import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"当前CUDA设备: {torch.cuda.current_device()}")
print(f"设备数量: {torch.cuda.device_count()}")
import sys
print(sys.executable)
import time
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import random
import os
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_sum, scatter_mean,scatter_softmax
import torch.nn.functional as F
from typing import List
from torch.optim import Adam
from tqdm import tqdm
from Bio import SeqIO

from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall
import geoopt
from src.config import *
from src.function import *
from src.model import *

config = Config()
data_config = config.data_config
train_config = config.train_config
seeding(config.seed)


train_file_list = os.listdir(config.data_config.train_npy_data_dir[:-6]+'seqs')
content_dict = {
    "pdb_id": [],
    "seq": []
}
for file in tqdm(train_file_list):
    if(file[-5:]!='fasta'):
        #print(file[-5:-1])
        continue
    sequences = read_fasta_biopython( os.path.join(data_config.train_fasta_data_dir , file) )
    content_dict["pdb_id"].append(list(sequences.keys())[0])
    content_dict["seq"].append(list(sequences.values())[0])

data = pd.DataFrame(content_dict)

# Split data into train, validation, and test sets
split = np.random.choice(['train', 'valid', 'test'], size=len(data), p=[0.8, 0.2, 0])
data['split'] = split
train_data = data[data['split']=='train']
valid_data = data[data['split']=='valid']
test_data = data[data['split']=='test']

os.makedirs(os.path.dirname(data_config.train_data_path), exist_ok=True)
train_data.to_csv(data_config.train_data_path, index=False)
valid_data.to_csv(data_config.valid_data_path, index=False)


train_dataset = RNADataset_without_ss_train(
    data_config.train_npy_data_dir,
    data_config.train_data_path,
)
valid_dataset = RNADataset_without_ss_train(
    data_config.valid_npy_data_dir,
    data_config.valid_data_path,
)

train_loader = DataLoader(train_dataset,
        batch_size=train_config.batch_size,
        shuffle= True,
        num_workers=0,
        collate_fn=featurize_without_ss)

valid_loader = DataLoader(valid_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=featurize_without_ss)
    
model = RNAModel_without_ss(config.model_config).to(config.device)
#print(model)

optimizer = Adam(model.parameters(), train_config.lr)
criterion = nn.CrossEntropyLoss()
if not os.path.exists(train_config.output_dir):
    os.makedirs(train_config.output_dir)
    
t_start = time.time()
save_recovery_list = []
save_loss_list = []
best_valid_recovery = 0
for epoch in range(train_config.epoch):
    model.train()
    epoch_loss = 0
    train_pbar = tqdm(train_loader)
    for batch in train_pbar:
        X, S, mask, lengths, names = batch
        X = perturb_coordinates(X, sigma=0.2)
        X = X.to(config.device)
        S = S.to(config.device)
        mask = mask.to(config.device)
        logits, S= model(X, S, mask)
        loss = criterion(logits, S)
        loss.backward()
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    save_loss_list.append(epoch_loss)
    print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, train_config.epoch, epoch_loss))
    
    model.eval()
    with torch.no_grad():
        recovery_list = []
        for batch in tqdm(valid_loader):
            X, S, mask, lengths, names = batch
            X = X.to(config.device)
            S = S.to(config.device)
            mask = mask.to(config.device)
            logits, S = model(X, S, mask)
            probs = F.softmax(logits, dim=-1)
            samples = probs.argmax(dim=-1)
            start_idx = 0
            for length in lengths:
                end_idx = start_idx + length.item()
                sample = samples[start_idx: end_idx]
                gt_S = S[start_idx: end_idx]
                arr = sample==gt_S
                recovery = (sample==gt_S).sum() / len(sample)
                recovery_list.append(recovery.cpu().numpy())
                start_idx = end_idx
        valid_recovery = np.mean(recovery_list)
        save_recovery_list.append(valid_recovery)
        print('Epoch {}/{}, recovery: {:.4f}'.format(epoch + 1, train_config.epoch, valid_recovery))
        if valid_recovery > best_valid_recovery:
            best_valid_recovery = valid_recovery
            torch.save(model.state_dict(), os.path.join(train_config.output_dir, 'HyperRdesign_without_ss.pt'))
#torch.save(model.state_dict(), 'RDesign_modle.pt')
t_end = time.time()
h, m, s = format_time_clean(t_end - t_start)
print(f"训练运行时间: {h:02d}:{m:02d}:{s:05.2f}")

loss_list = np.array(save_loss_list)
np.save(data_config.track_loss_path, loss_list)
best_valid_recovery = np.array(save_recovery_list)
np.save(data_config.track_recovery_path, best_valid_recovery)
#loaded = np.load("array.npy")  # 读取
