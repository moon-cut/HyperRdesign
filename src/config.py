from dataclasses import dataclass, field
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from typing import List

@dataclass
class DataConfig:
    train_npy_data_dir: str = './train_data/coords'
    train_fasta_data_dir: str = './train_data/seqs'
    train_data_path: str = './temp/public_train_data.csv'
    train_ss_data_dir: str = './train_data/SS/'

    valid_npy_data_dir: str = './train_data/coords'
    valid_data_path: str = './temp/public_valid_data.csv'
    valid_ss_data_dir: str = './train_data/SS/'

    test_npy_data_dir: str = './test_data/coords'
    test_ss_data_dir: str = './test_data/SS/'
    test_output_data_dir: str = './test_data/output_seqs/'

    track_loss_path: str = './temp/loss_list.npy'
    track_recovery_path: str = './temp/valid_recovery_list.npy'

@dataclass
class ModelConfig:
    smoothing: float = 0.05
    hidden: int = 128
    vocab_size: int = 4  # 明确指定为 int 类型
    k_neighbors: int = 30  # 明确指定为 int 类型
    dropout: float = 0.05
    node_feat_types: List[str] = field(default_factory=lambda: ['angle', 'distance', 'direction'])  # 使用 field 避免可变对象问题
    edge_feat_types: List[str] = field(default_factory=lambda: ['orientation', 'distance', 'direction'])  # 同上
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3  # 修正为整数，去掉多余的小数点

@dataclass
class TrainConfig:
    batch_size: int = 16
    epoch: int = 150
    lr: float = 0.001
    output_dir: str = 'ckpts'
    ckpt_path: str = 'ckpts/HyperRdesign.pt'
    ckpt_path_without_ss: str = 'ckpts/HyperRdesign_without_ss.pt'

@dataclass
class Config:
    pipeline: str = 'train'
    seed: int = 2025
    device: str = 'cuda:0' 
    data_config: DataConfig = field(default_factory=lambda: DataConfig())
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    train_config: TrainConfig = field(default_factory=TrainConfig)


# Define RNADataset Class and Seeding Function
class RNADataset_test(Dataset):
    def __init__(self, npy_dir,ss_dir):
        super(RNADataset_test, self).__init__()
        coords_file = os.listdir(npy_dir)
        self.name_list = [s[:-4] for s in coords_file]

        self.npy_dir = npy_dir
        self.ss_dir = ss_dir

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))
        ss_infor = np.load(os.path.join(self.ss_dir, pdb_id + '.npy'))

        feature = {
            "name": pdb_id,
            "ss_infor" : ss_infor,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }

        return feature

# Define RNADataset Class and Seeding Function
class RNADataset(Dataset):
    def __init__(self, npy_dir,ss_dir, *data_path):
        super(RNADataset, self).__init__()
        self.data_path = data_path
        if len(self.data_path)>0:
            self.data = pd.read_csv(data_path[0])
            self.npy_dir = npy_dir
            self.ss_dir = ss_dir
            self.seq_list = self.data['seq'].to_list()
            self.name_list = self.data['pdb_id'].to_list()
        else:
            coords_file = os.listdir(npy_dir)
            self.name_list = [s[:-4] for s in coords_file]
            self.npy_dir = npy_dir
            self.ss_dir = ss_dir

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))
        ss_infor = np.load(os.path.join(self.ss_dir, pdb_id + '.npy'))

        feature = {
            "name": pdb_id,
            "ss_infor" : ss_infor,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }
        if len(self.data_path)>0:
            seq = self.seq_list[idx]
            feature["seq"]=seq

        return feature
    
# Define RNADataset Class and Seeding Function
class RNADataset_without_ss_train(Dataset):
    def __init__(self, npy_dir, *data_path):
        super(RNADataset_without_ss_train, self).__init__()
        self.data_path=data_path
        if len(self.data_path)>0:
            self.data = pd.read_csv(data_path[0])
            self.npy_dir = npy_dir
            self.seq_list = self.data['seq'].to_list()
            self.name_list = self.data['pdb_id'].to_list()      
        else:
            coords_file = os.listdir(npy_dir)
            self.name_list = [s[:-4] for s in coords_file]
            self.npy_dir = npy_dir  

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))

        feature = {
            "name": pdb_id,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }
        if len(self.data_path)>0:
            seq = self.seq_list[idx]
            feature["seq"]=seq
        return feature