import itertools
import sys
import os

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

from collections import defaultdict
from operator import neg
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

df_drugs_smiles = pd.read_csv(root_path + '/drugbank_test/drugbank/drug_smiles.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in
                         zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

drug_to_mol_graph = {id: Chem.MolFromSmiles(smiles.strip()) for id, smiles in
                     zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])}

# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list(
    {a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list(
    {a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list(
    {a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(
    a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max(
    [a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in
                              itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# 获取边特征
def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def atom_features(atom,
                  explicit_H=True,
                  use_chirality=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
         'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
         'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
         ]) + [atom.GetDegree() / 10, atom.GetImplicitValence(),
               atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):
    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature


def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    # 这里是之前的
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list

    # 新的数据处理过程
    # 添加了边特征*edge_features(b)) for b in mol_graph.GetBonds()
    new_edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    # Separate (bond_i, bond_j, bond_features) to (bond_i, bond_j) and bond_features
    new_edge_list, edge_feats = (new_edge_list[:, :2], new_edge_list[:, 2:].float()) if len(new_edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    new_edge_list = torch.cat([new_edge_list, new_edge_list[:, [1, 0]]], dim=0) if len(new_edge_list) else new_edge_list
    # 获取edge_feats，参考SADDI
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
    # line_edge_index
    # 这里很关键，DMPNN中要求有line_graph_edge_index，这段代码参考SADDI
    line_graph_edge_index = torch.LongTensor([])
    if new_edge_list.nelement() != 0:
        conn = (new_edge_list[:, 1].unsqueeze(1) == new_edge_list[:, 0].unsqueeze(0)) & (
                new_edge_list[:, 0].unsqueeze(1) != new_edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    return new_edge_list.T, n_features, edge_feats, line_graph_edge_index


def get_bipartite_graph(mol_graph_1, mol_graph_2):
    x1 = np.arange(0, len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0, len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1, x2))
    edge_list = torch.stack([edge_list[0].reshape(-1), edge_list[1].reshape(-1)])
    return edge_list


MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol)
                          for drug_id, mol in drug_id_mol_graph_tup}

MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])

##### DDI statistics and counting #######
df_all_pos_ddi = pd.read_csv(root_path + '/drugbank_test/drugbank/ddis.csv')
all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type'])]

ALL_DRUG_IDS, _ = zip(*drug_id_mol_graph_tup)
ALL_DRUG_IDS = np.array(list(set(ALL_DRUG_IDS)))
ALL_TRUE_H_WITH_TR = defaultdict(list)
ALL_TRUE_T_WITH_HR = defaultdict(list)

FREQ_REL = defaultdict(int)
ALL_H_WITH_R = defaultdict(dict)
ALL_T_WITH_R = defaultdict(dict)
ALL_TAIL_PER_HEAD = {}
ALL_HEAD_PER_TAIL = {}

for h, t, r in all_pos_tup:
    ALL_TRUE_H_WITH_TR[(t, r)].append(h)
    ALL_TRUE_T_WITH_HR[(h, r)].append(t)
    FREQ_REL[r] += 1.0
    ALL_H_WITH_R[r][h] = 1
    ALL_T_WITH_R[r][t] = 1

for t, r in ALL_TRUE_H_WITH_TR:
    ALL_TRUE_H_WITH_TR[(t, r)] = np.array(list(set(ALL_TRUE_H_WITH_TR[(t, r)])))
for h, r in ALL_TRUE_T_WITH_HR:
    ALL_TRUE_T_WITH_HR[(h, r)] = np.array(list(set(ALL_TRUE_T_WITH_HR[(h, r)])))

for r in FREQ_REL:
    ALL_H_WITH_R[r] = np.array(list(ALL_H_WITH_R[r].keys()))
    ALL_T_WITH_R[r] = np.array(list(ALL_T_WITH_R[r].keys()))
    ALL_HEAD_PER_TAIL[r] = FREQ_REL[r] / len(ALL_T_WITH_R[r])
    ALL_TAIL_PER_HEAD[r] = FREQ_REL[r] / len(ALL_H_WITH_R[r])


#######    ****** ###############
class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class DrugDataset(Dataset):
    def __init__(self, tri_list, ratio=1.0, neg_ent=1, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        '''
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio

        for h, t, r, *_ in tri_list:
            if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((h, t, r))
        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1 + d2)))
        else:
            self.drug_ids = ALL_DRUG_IDS

        self.drug_ids = np.array([id for id in self.drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []
        pos_b_samples = []

        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []
        neg_b_samples = []

        neg_h_test = []
        neg_t_test = []

        for h, t, r in batch:
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            h_graph = drug_to_mol_graph[h]
            t_graph = drug_to_mol_graph[t]

            pos_b_graph = self._create_b_graph(get_bipartite_graph(h_graph, t_graph), h_data.x, t_data.x)

            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)
            pos_b_samples.append(pos_b_graph)

            neg_heads, neg_tails = self.__normal_batch(h, t, r, self.neg_ent)

            for neg_h in neg_heads:
                neg_rels.append(r)
                neg_h_samples.append(self.__create_graph_data(neg_h))
                neg_t_samples.append(t_data)

                neg_h_test.append(neg_h)
                neg_t_test.append(t)

            for neg_t in neg_tails:
                neg_rels.append(r)
                neg_h_samples.append(h_data)
                neg_t_samples.append(self.__create_graph_data(neg_t))

                neg_h_test.append(h)
                neg_t_test.append(neg_t)

        for i in range(len(neg_h_test)):
            neg_h_graph = drug_to_mol_graph[neg_h_test[i]]
            neg_t_graph = drug_to_mol_graph[neg_t_test[i]]
            neg_b_graph = self._create_b_graph(get_bipartite_graph(neg_h_graph, neg_t_graph), neg_h_samples[i].x,
                                               neg_t_samples[i].x)
            neg_b_samples.append(neg_b_graph)

        # 因为DMPNN中要求输入edge_index_batch
        # 所以这里需要添加follow_batch=['edge_index']
        pos_h_samples = Batch.from_data_list(pos_h_samples, follow_batch=['edge_index'])
        pos_t_samples = Batch.from_data_list(pos_t_samples, follow_batch=['edge_index'])
        pos_b_samples = Batch.from_data_list(pos_b_samples, follow_batch=['edge_index'])
        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)

        pos_tri = (pos_h_samples, pos_t_samples, pos_rels, pos_b_samples)

        # 同上
        neg_h_samples = Batch.from_data_list(neg_h_samples, follow_batch=['edge_index'])
        neg_t_samples = Batch.from_data_list(neg_t_samples, follow_batch=['edge_index'])
        neg_b_samples = Batch.from_data_list(neg_b_samples, follow_batch=['edge_index'])
        neg_rels = torch.LongTensor(neg_rels).unsqueeze(0)
        neg_tri = (neg_h_samples, neg_t_samples, neg_rels, neg_b_samples)

        return pos_tri, neg_tri

    def __create_graph_data(self, id):
        # get_mol_edge_list_and_feat_mtx最后返回了四个值
        # 依次是边，节点特征，边特征，line_graph_edge_index
        edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
        n_features = MOL_EDGE_LIST_FEAT_MTX[id][1]
        edge_attr = MOL_EDGE_LIST_FEAT_MTX[id][2]
        line_graph_edge_index = MOL_EDGE_LIST_FEAT_MTX[id][3]
        # 构建图
        data = CustomData(x=n_features,
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          line_graph_edge_index=line_graph_edge_index)

        return data

    def _create_b_graph(self, edge_index, x_s, x_t):
        return BipartiteData(edge_index, x_s, x_t)

    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])

        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])

    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, ALL_TRUE_T_WITH_HR, n)

    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        return (self.__corrupt_head(t, r, neg_size_h),
                self.__corrupt_tail(h, r, neg_size_t))


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''

    def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"
