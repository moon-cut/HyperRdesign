import torch
import sys
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
from src.function import _normalize





feat_dims = {
    'node': {
        'angle': 12,
        'distance': 80,
        'direction': 9,
    },
    'edge': {
        'orientation': 4,
        'distance': 96,
        'direction': 15,
    }
}

# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_hidden + num_in, num_heads)
        #self.attention = NeighborAttention(num_hidden, num_hidden, num_heads)
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id=None):
        center_id = edge_idx[0]
        dh = self.attention(h_V, h_E, center_id, batch_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V
# NeighborAttention
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        #self.W_K = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden*3, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        Q = self.W_Q(h_V).view(N, n_heads, 1, d)[center_id]
        K = self.W_K(h_E).view(E, n_heads, d, 1)
        attend_logits = torch.matmul(Q, K).view(E, n_heads, 1)
        attend_logits = attend_logits / np.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])
        h_V_update = self.W_O(h_V)
        return h_V_update


class LearnablePoincareBall(nn.Module):
    def __init__(self, init_c=1.0):
        super().__init__()
        self.log_c = nn.Parameter(torch.log(torch.tensor(init_c)))

    @property
    def c(self):
        return torch.exp(self.log_c)

    def projx(self, x):
        # 直接实现投影逻辑，不用修改 geoopt
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-5) / torch.sqrt(self.c)
        scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-15), max=1.0)
        return x * scale

    def expmap0(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(self.c)
        return torch.tanh(sqrt_c * norm) * x / (norm * sqrt_c + 1e-15)

    def logmap0(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(self.c)
        return (1.0 / sqrt_c) * torch.atanh(torch.clamp(sqrt_c * norm, max=1 - 1e-5)) * x / (norm + 1e-15)

    def mobius_add(self, x, y):
        # 简化版本，适合小批量
        x2 = (x**2).sum(dim=-1, keepdim=True)
        y2 = (y**2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        c = self.c
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        return num / (denom + 1e-15)

class HyperbolicMPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, manifold=None, dropout=0.1, init_c=1.0):
        super(HyperbolicMPNNLayer, self).__init__()
        self.manifold = manifold or LearnablePoincareBall(init_c=init_c)
        self.dropout = nn.Dropout(dropout)

        self.W1 = nn.Linear(num_hidden*5, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)

    def mobius_linear(self, x, W):
        tangent = self.manifold.logmap0(x)
        out = W(tangent)
        return self.manifold.expmap0(out)
    
    def hyperbolic_aggregate(self, h_message, src, manifold, dim=0):
        tangent = manifold.logmap0(h_message)
        agg_tangent = scatter_mean(tangent, src, dim=dim)
        return manifold.expmap0(agg_tangent)

    def forward(self, h_V, h_E, edge_idx, batch_id=None):
        src, dst = edge_idx

        h_E = self.manifold.projx(h_E)
        h_V = self.manifold.projx(h_V)

        # 消息传播
        h_EV = self.mobius_linear(torch.cat([h_E, h_V[src], h_V[dst]], dim=-1), self.W1)
        h_EV = self.mobius_linear(self.manifold.projx(F.relu(h_EV)), self.W2)
        h_message = self.mobius_linear(self.manifold.projx(F.relu(h_EV)), self.W3)

        agg = self.hyperbolic_aggregate(h_message, src, self.manifold, dim=0)
        h_V = self.manifold.mobius_add(h_V, self.dropout(agg))
        h_V = self.manifold.projx(h_V)
        # print(f"[Layer Curvature] c = {self.manifold.c.item():.4f}")

        return h_V



class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.ReLU()

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id=None):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_E)))))
        dh = scatter_sum(h_message, src_idx, dim=0, dim_size=h_V.shape[0]) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        return h_V

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class RNAFeatures(nn.Module):
    def __init__(self, edge_features, node_features, node_feat_types=[], edge_feat_types=[], num_rbf=16, top_k=30, augment_eps=0., dropout=0.1, args=None):
        super(RNAFeatures, self).__init__()
        """Extract RNA Features"""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout)
        self.node_feat_types = node_feat_types
        self.edge_feat_types = edge_feat_types

        node_in = sum([feat_dims['node'][feat] for feat in node_feat_types])
        edge_in = sum([feat_dims['edge'][feat] for feat in edge_feat_types])
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)
        
    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx
    
    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    
    def _get_rbf(self, A, B, E_idx=None, num_rbf=16):
        if E_idx is not None:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
            D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0]
            RBF_A_B = self._rbf(D_A_B_neighbors)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,:,None,:])**2,-1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B
    
    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q
    
    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        V = X.clone()
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3) 
        dX = X[:,1:,:] - X[:,:-1,:]
        U = _normalize(dX, dim=-1)
        u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
        n_0 = _normalize(torch.cross(u_0, u_1, dim=-1), dim=-1)
        b_1 = _normalize(u_0 - u_1, dim=-1)
        
        # select C3'
        n_0 = n_0[:,4::6,:] 
        b_1 = b_1[:,4::6,:]
        X = X[:,4::6,:]

        Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0, dim=-1)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9])
        Q = F.pad(Q, (0,0,0,1), 'constant', 0) # [16, 464, 9]

        Q_neighbors = gather_nodes(Q, E_idx) # [16, 464, 30, 9]
        P_neighbors = gather_nodes(V[:,:,0,:], E_idx) # [16, 464, 30, 3]
        O5_neighbors = gather_nodes(V[:,:,1,:], E_idx)
        C5_neighbors = gather_nodes(V[:,:,2,:], E_idx)
        C4_neighbors = gather_nodes(V[:,:,3,:], E_idx)
        O3_neighbors = gather_nodes(V[:,:,5,:], E_idx)
        
        Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
        Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

        dX = torch.stack([P_neighbors,O5_neighbors,C5_neighbors,C4_neighbors,O3_neighbors], dim=3) - X[:,:,None,None,:] # [16, 464, 30, 3]
        dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = _normalize(dU, dim=-1)
        E_direct = E_direct.reshape(B, N, K,-1)
        R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
        E_orient = self._quaternions(R)
        
        dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
        dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
        dU_inner = _normalize(dU_inner, dim=-1)
        V_direct = dU_inner.reshape(B,N,-1)
        return V_direct, E_direct, E_orient
    
    def _dihedrals(self, X, eps=1e-7):
        # P, O5', C5', C4', C3', O3'
        X = X[:,:,:6,:].reshape(X.shape[0], 6*X.shape[1], 3)

        # Shifted slices of unit vectors
        # https://iupac.qmul.ac.uk/misc/pnuc2.html#220
        # https://x3dna.org/highlights/torsion-angles-of-nucleic-acid-structures
        # alpha:   O3'_{i-1} P_i O5'_i C5'_i
        # beta:    P_i O5'_i C5'_i C4'_i
        # gamma:   O5'_i C5'_i C4'_i C3'_i
        # delta:   C5'_i C4'_i C3'_i O3'_i
        # epsilon: C4'_i C3'_i O3'_i P_{i+1}
        # zeta:    C3'_i O3'_i P_{i+1} O5'_{i+1} 
        # What's more:
        #   chi: C1' - N9 
        #   chi is different for (C, T, U) and (A, G) https://x3dna.org/highlights/the-chi-x-torsion-angle-characterizes-base-sugar-relative-orientation

        dX = X[:, 5:, :] - X[:, :-5, :] # O3'-P, P-O5', O5'-C5', C5'-C4', ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]  # O3'-P, P-O5', ...
        u_1 = U[:,1:-1,:] # P-O5', O5'-C5', ...
        u_0 = U[:,2:,:]   # O5'-C5', C5'-C4', ...
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        
        D = F.pad(D, (3,4), 'constant', 0)
        D = D.view((D.size(0), D.size(1) //6, 6))
        return torch.cat((torch.cos(D), torch.sin(D)), 2) # return D_features
    
    def forward(self, X, S, mask):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        B, N, _,_ = X.shape
        # P, O5', C5', C4', C3', O3'
        atom_P = X[:, :, 0, :]
        atom_O5_ = X[:, :, 1, :]
        atom_C5_ = X[:, :, 2, :]
        atom_C4_ = X[:, :, 3, :]
        atom_C3_ = X[:, :, 4, :] 
        atom_O3_ = X[:, :, 5, :]

        X_backbone = atom_P
        D_neighbors, E_idx = self._dist(X_backbone, mask)        

        mask_bool = (mask==1)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # node features
        h_V = []
        # angle
        V_angle = node_mask_select(self._dihedrals(X))
        # distance
        node_list = ['O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        V_dist = []
        
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            V_dist.append(node_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(V_dist), dim=-1).squeeze()
        # direction
        V_direct, E_direct, E_orient = self._orientations_coarse(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct, E_orient = list(map(lambda x: edge_mask_select(x), [E_direct, E_orient]))

        # edge features
        h_E = []
        # dist
        edge_list = ['P-P', 'O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        E_dist = [] 
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_dist.append(edge_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)))
        E_dist = torch.cat(tuple(E_dist), dim=-1)

        if 'angle' in self.node_feat_types:
            h_V.append(V_angle)
        if 'distance' in self.node_feat_types:
            h_V.append(V_dist)
        if 'direction' in self.node_feat_types:
            h_V.append(V_direct)

        if 'orientation' in self.edge_feat_types:
            h_E.append(E_orient)
        if 'distance' in self.edge_feat_types:
            h_E.append(E_dist)
        if 'direction' in self.edge_feat_types:
            h_E.append(E_direct)
            
        # Embed the nodes
        h_V = self.norm_nodes(self.node_embedding(torch.cat(h_V, dim=-1)))
        h_E = self.norm_edges(self.edge_embedding(torch.cat(h_E, dim=-1)))

        # prepare the variables to return
        S = torch.masked_select(S, mask_bool)
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]
        return X, S, h_V, h_E, E_idx, batch_id
    
class RNAModel(nn.Module):
    def __init__(self, model_config):
        super(RNAModel, self).__init__()

        self.smoothing = model_config.smoothing
        self.node_features = self.edge_features = model_config.hidden
        self.hidden_dim = model_config.hidden
        self.vocab = model_config.vocab_size

        self.features = RNAFeatures(
            model_config.hidden, model_config.hidden, 
            top_k=model_config.k_neighbors, 
            dropout=model_config.dropout,
            node_feat_types=model_config.node_feat_types, 
            edge_feat_types=model_config.edge_feat_types,
            args=model_config
        )

        
        layer0 = HyperbolicMPNNLayer
        layer1 = TransformerLayer #MPNNLayer
        layer = MPNNLayer

        self.W_s = nn.Embedding(model_config.vocab_size, self.hidden_dim)

        self.ss_encoder_layers = nn.ModuleList([
            layer(5, 5*2, dropout=model_config.dropout)
            for _ in range(model_config.num_decoder_layers)])

        self.encoder_layers = nn.ModuleList([
            layer0(self.hidden_dim, self.hidden_dim*2, dropout=model_config.dropout)
            for _ in range(model_config.num_encoder_layers)])
        
        self.decoder_layers = nn.ModuleList([
            layer1(self.hidden_dim, (self.hidden_dim)*2, dropout=model_config.dropout)   # 5 是 ss_node_feats的维度
            for _ in range(model_config.num_decoder_layers)])       

        self.feature_reshape0 = nn.Linear(128+5, 128, bias=True)
        self.feature_reshape1 = nn.ReLU()

        self.readout = nn.Linear(self.hidden_dim, model_config.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, ss_info_list):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)
        ss_node_feats, ss_edge_feats, ss_edge_index = ss_feature(S, mask, ss_info_list) # 已经处理完特征，接下来将其融合到h_V和h_E中  add bu liu 0930

        for ss_enc_layer in self.ss_encoder_layers:
            ss_h_EV = torch.cat([ss_edge_feats, ss_node_feats[ss_edge_index[0]], ss_node_feats[ss_edge_index[1]]], dim=-1)
            ss_node_feats = ss_enc_layer(ss_node_feats, ss_h_EV, ss_edge_index, batch_id)

        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        h_V =  torch.cat([h_V, ss_node_feats], dim=-1)  # 融合节点特征  add by liu 0930
        h_V = self.feature_reshape0(h_V)
        h_V = self.feature_reshape1(h_V)
        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        logits = self.readout(h_V)
        return logits, S

    def sample(self, X, S, mask=None):
        X, gt_S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        logits = self.readout(h_V)
        return logits, gt_S

class RNAModel_without_ss(nn.Module):
    def __init__(self, model_config):
        super(RNAModel_without_ss, self).__init__()

        self.smoothing = model_config.smoothing
        self.node_features = self.edge_features = model_config.hidden
        self.hidden_dim = model_config.hidden
        self.vocab = model_config.vocab_size

        self.features = RNAFeatures(
            model_config.hidden, model_config.hidden, 
            top_k=model_config.k_neighbors, 
            dropout=model_config.dropout,
            node_feat_types=model_config.node_feat_types, 
            edge_feat_types=model_config.edge_feat_types,
            args=model_config
        )

        
        layer0 = HyperbolicMPNNLayer
        layer1 = TransformerLayer #MPNNLayer

        self.W_s = nn.Embedding(model_config.vocab_size, self.hidden_dim)

        self.encoder_layers = nn.ModuleList([
            layer0(self.hidden_dim, self.hidden_dim*2, dropout=model_config.dropout)
            for _ in range(model_config.num_encoder_layers)])
        
        self.decoder_layers = nn.ModuleList([
            layer1(self.hidden_dim, (self.hidden_dim)*2, dropout=model_config.dropout)   # 5 是 ss_node_feats的维度
            for _ in range(model_config.num_decoder_layers)])       

        self.readout = nn.Linear(self.hidden_dim, model_config.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)
        #ss_node_feats, ss_edge_feats, ss_edge_index = ss_feature(S, mask) # 已经处理完特征，接下来将其融合到h_V和h_E中  add bu liu 0930

        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        logits = self.readout(h_V)
        return logits, S

    def sample(self, X, S, mask=None):
        X, gt_S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        for enc_layer in self.encoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = enc_layer(h_V, h_EV, E_idx, batch_id)

        for dec_layer in self.decoder_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = dec_layer(h_V, h_EV, E_idx, batch_id)

        logits = self.readout(h_V)
        return logits, gt_S

