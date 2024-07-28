from datetime import datetime
import time
import argparse
import torch

# from signal import signal, SIGPIPE, SIG_IGN
# signal(SIGPIPE, SIG_IGN)

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# default batch size = 1024

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_drugbank.pkl')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)


############################################################
###### Dataset
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_tup]
    val_tup = [(tup[0], tup[1], int(tup[2])) for tup in val_tup]

    return train_tup, val_tup


df_ddi_train = pd.read_csv('drugbank/fold0/train.csv')  ## 修改fold0/1/2就可以进行交叉验证，将三次结果取平均
df_ddi_test = pd.read_csv('drugbank/fold0/test.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
train_tup, val_tup = split_train_valid(train_tup, 2, val_ratio=0.2)
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

# 这里改回1
ratio = 1  # ration改为1则表示使用全部数据
train_tup = train_tup[:int(len(train_tup) * ratio)]
val_tup = val_tup[:int(len(val_tup) * ratio)]
test_tup = test_tup[:int(len(test_tup) * ratio)]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

num_workers = 0
train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3, num_workers=num_workers)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3, num_workers=num_workers)


def do_compute(batch, device, model):
    '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    max_acc = 0
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            # loss = loss + reg_loss(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = do_compute_metrics(
                train_probas_pred, train_ground_truth)

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(
                val_probas_pred, val_ground_truth)
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)

        if scheduler:
            # print('scheduling')
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
              f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(
            f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')


def test(test_data_loader, model):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = do_compute_metrics(
            test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(
        f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')


# model = models.CSS_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
model = models.MVN_DDI(n_atom_feats,
                       n_atom_hid,
                       kge_dim,
                       rel_total,
                       heads_out_feat_params=[64, 64],
                       blocks_params=[2, 2])
loss = custom_loss.SigmoidLoss()
# reg_loss = Regularization(model, weight_decay=0.01, p=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
print(model)
model.to(device=torch.device(device))
# # if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)
test_model = torch.load(pkl_name)
test(test_data_loader, test_model)
