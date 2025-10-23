import torch
import random
import os
import numpy as np
from Bio import SeqIO

# Define function to read FASTA files using Biopython
def read_fasta_biopython(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences
# Optional: from torch_ geometric.data import Data

def perturb_coordinates(X, sigma=0.1):
    noise = torch.randn_like(X) * sigma
    return X + noise


#从点括号中提取碱基对
def parse_dot_bracket(dot: str):
    """
    Parse dot-bracket string into pair_map: list of length L, where pair_map[i] = j (0-indexed) or -1
    Supports simple pseudoknot notation? Standard dot-bracket only supports nested pairs. 
    For pseudoknots you'd need extended notation (e.g. [], {}, <>) — here we support basic multiple bracket types.
    """
    L = len(dot)
    pair_map = [-1]*L

    # bracket types stack: '() [] {} <>'
    pairs = {'(':')', '[':']', '{':'}', '<':'>'}
    opens = {o:[] for o in pairs.keys()}
    closers = {v:k for k,v in pairs.items()}

    for i, ch in enumerate(dot):
        if ch in opens:
            opens[ch].append(i)
        elif ch in closers:
            open_ch = closers[ch]
            if len(opens[open_ch])==0:
                print("ERROR")
                # unmatched closer -> ignore or raise
                continue
            j = opens[open_ch].pop()
            pair_map[i] = j
            pair_map[j] = i
        else:
            # dot or other char => unpaired
            continue

    return pair_map  # -1 for unpaired, else partner index

# helper detect stems and loops (simple)
def extract_stems_from_pairmap(pair_map):
    """
    Return a list of stems (each stem is list of paired index pairs [(i,j),...])
    This is a simple greedy scan: contiguous consecutive basepairs form stems.
    """
    L = len(pair_map)
    visited = [False]*L
    stems = []
    for i in range(L):
        if pair_map[i] == -1 or visited[i]:
            continue
        j = pair_map[i]
        if j <= i: continue
        # try extend as stem: check i+1 paired with j-1 etc.
        stem = []
        a, b = i, j
        while a < b and pair_map[a]==b:
            stem.append((a,b))
            visited[a] = visited[b] = True
            a += 1; b -= 1
            if a>=len(pair_map) or b<0: break
        stems.append(stem)
    return stems

def compute_tree_depths_from_pairs(pair_map):
    """
    Build a simple 'nested' tree based on pair nesting: each base-pair corresponds to a node in tree.
    For simplicity produce per-residue depth = number of enclosing pairs.
    (This ignores pseudoknots — they break proper nesting.)
    """
    L = len(pair_map)
    depth = [0]*L
    stack = []
    for i in range(L):
        if pair_map[i] > i:  # opening of a nested pair
            depth[i] = len(stack)
            stack.append(i)
        elif pair_map[i] == -1:
            depth[i] = len(stack)
        else:
            # closing or paired to earlier
            depth[i] = len(stack)-1 if len(stack)>0 else 0
            if stack and pair_map[i] == stack[-1]:
                stack.pop()
    # clamp depth to non-neg
    depth = [max(0,int(d)) for d in depth]
    return depth

def build_secondary_graph(dotbracket, include_stack_edges=True, include_pseudoknot=True):
    """
    Build graph from sequence + dotbracket.
    Returns:
      node_feats: torch.Tensor (L, node_feat_dim)
      edge_index: torch.LongTensor (2, E)
      edge_attr: torch.Tensor (E, edge_feat_dim)
      meta: dict with pair_map, mask, stems, depth, etc.
    Edge types encoded as one-hot: [backbone, basepair, pseudoknot, stack]
    """
    #assert len(seq) == len(dotbracket)
    L = len(dotbracket)
    pair_map = parse_dot_bracket(dotbracket)  # -1 or partner idx

    # Node features:
    # - base one-hot (5)
    # - is_paired (1)
    # - paired_distance = abs(i - partner) (1, normalized later)
    # - pos_norm (1)   - position / L
    # - depth (1)      - number of enclosing pairs
    # - degree (1)     - number neighbors in sec graph
    # Total node feat dim ~ 10


    is_paired = np.array([0.0 if p==-1 else 1.0 for p in pair_map], dtype=np.float32)
    paired_dist = np.array([abs(i-pair_map[i]) if pair_map[i]!=-1 else 0 for i in range(L)], dtype=np.float32)
    pos_norm = np.array([i/(L-1) if L>1 else 0.0 for i in range(L)], dtype=np.float32)
    depth = np.array(compute_tree_depths_from_pairs(pair_map), dtype=np.float32)

    # Build adjacency edges
    edge_list = []
    edge_attr_list = []

    # helper to add edge (i->j and j->i if undirected; keep directed if desired)
    def add_edge(i, j, etype_onehot, extra_attr=None):
        edge_list.append((i,j))
        # edge_attr: concatenation of etype_onehot + extra numeric features (e.g., seq_dist)
        seq_dist = float(abs(i-j))
        if extra_attr is None:
            extra_attr = []
        edge_attr_list.append(np.concatenate([etype_onehot, np.array([seq_dist], dtype=np.float32), np.array(extra_attr, dtype=np.float32)]))

    # edge type one-hot mapping
    ET_BACKBONE = np.array([1,0,0,0], dtype=np.float32)
    ET_BASEPAIR = np.array([0,1,0,0], dtype=np.float32)
    ET_PSEUDOKNOT = np.array([0,0,1,0], dtype=np.float32)
    ET_STACK = np.array([0,0,0,1], dtype=np.float32)

    # 1) backbone edges (i <-> i+1)
    for i in range(L-1):
        add_edge(i, i+1, ET_BACKBONE)
        add_edge(i+1, i, ET_BACKBONE)

    # 2) basepair edges
    for i in range(L):
        j = pair_map[i]
        if j != -1 and j > i:
            # add both directions
            add_edge(i, j, ET_BASEPAIR)
            add_edge(j, i, ET_BASEPAIR)

    # 3) stack edges (optional): connect adjacent paired bases along stems (i paired with j, i+1 paired with j-1 => stacking)
    if include_stack_edges:
        for i in range(L-1):
            j = pair_map[i]
            k = pair_map[i+1]
            if j != -1 and k != -1 and j - 1 == k:  # i paired with j, i+1 paired with j-1
                add_edge(i, i+1, ET_STACK)
                add_edge(i+1, i, ET_STACK)
                add_edge(j, j-1, ET_STACK)
                add_edge(j-1, j, ET_STACK)

    # 4) pseudoknots: detection is non-trivial. We detect crossing pairs: (i,j) and (p,q) s.t. i<p<j<q
    #    mark the crossing edges as pseudoknot edges (if include_pseudoknot True)
    if include_pseudoknot:
        pairs = [(i, pair_map[i]) for i in range(L) if pair_map[i] > i]
        for a,b in pairs:
            for c,d in pairs:
                if (a < c < b < d) or (c < a < d < b):
                    # crossing detected -> mark edges (a,b) and (c,d) as pseudoknot-type in addition to basepair
                    # We add pseudoknot-specific directed edges as well.
                    add_edge(a, b, ET_PSEUDOKNOT)
                    add_edge(b, a, ET_PSEUDOKNOT)
                    add_edge(c, d, ET_PSEUDOKNOT)
                    add_edge(d, c, ET_PSEUDOKNOT)

    # convert node features to torch tensor
    node_feats = np.concatenate([
        #base_oh,
        is_paired.reshape(-1,1),
        paired_dist.reshape(-1,1),
        pos_norm.reshape(-1,1),
        depth.reshape(-1,1)
    ], axis=1).astype(np.float32)

    # compute degree (count distinct neighbor types) and append
    deg = np.zeros((L,1), dtype=np.float32)
    for (i,j) in edge_list:
        deg[i,0] += 1.0
    node_feats = np.concatenate([node_feats, deg], axis=1)

    # normalize paired_dist (cap by L, then /L)
    node_feats[:, 1] = node_feats[:,1] / max(1.0, L)  # assuming paired_dist at index 6 if base_oh(5) + is_paired(1) + paired_dist(1) => indices vary

    # convert edges to tensors
    if len(edge_list)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, 1+4), dtype=torch.float32)
    else:
        ei = np.array(edge_list, dtype=np.int64).T  # (2, E)
        ea = np.stack(edge_attr_list, axis=0)       # (E, feat)
        edge_index = torch.from_numpy(ei).long()
        edge_attr = torch.from_numpy(ea).float()

    data = {
        "node_feats": torch.from_numpy(node_feats).float(),  # (L, D)
        "edge_index": edge_index,   # (2, E)
        "edge_attr": edge_attr,     # (E, edge_feat_dim)
        "pair_map": pair_map,
        "depth": depth.tolist(),
        "stems": extract_stems_from_pairmap(pair_map),
        "mask": torch.ones(L, dtype=torch.bool)
    }

    return data

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seeding done!!!')

def format_time_clean(seconds):
    """使用divmod格式化时间"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds

def featurize(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['coords']['P']) for b in batch], dtype=np.int32)
    L_max = max([len(b['coords']['P']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    names = []
    ss_info_list = []

    # Build the batch
    for i, b in enumerate(batch):
        #x = np.stack([b['coords'][c] for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        x = np.stack([np.nan_to_num(b['coords'][c], nan=0.0) for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        l = len(b['coords']['P'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad
        if 'seq' in b:
            indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
            S[i, :l] = indices
        names.append(b['name'])
        #print(b['ss_infor'])
        if 'ss_infor' in b:
            ss_info_i = build_secondary_graph(str(b['ss_infor']))
            ss_info_list.append(ss_info_i)
        
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    #mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    if 'ss_infor' in b:
        return X, S, mask, lengths, names, ss_info_list
    else:
        return X, S, mask, lengths, names



def featurize_without_ss(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        #x = np.stack([b['coords'][c] for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        x = np.stack([np.nan_to_num(b['coords'][c], nan=0.0) for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        names.append(b['name'])
        
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    #mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths, names


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def ss_feature(S, mask, ss_info_list):
    #print(S.shape)

    ss_node_feats = []
    # 将每个二维数组转换为张量并添加到列表中
    for i in range(len(ss_info_list)):
        ss_node_feats.append(ss_info_list[i]["node_feats"])
    ss_node_feats = torch.cat(ss_node_feats, dim=0)

    ss_edge_feats=[]
    for i in range(len(ss_info_list)):
        ss_edge_feats.append(ss_info_list[i]["edge_attr"])
    ss_edge_feats = torch.cat(ss_edge_feats, dim=0)
    #print(ss_edge_feats.shape)

    ss_edge_index=[]
    index = 0
    for i in range(len(ss_info_list)):
        edge_index = ss_info_list[i]["edge_index"] + index*torch.ones(ss_info_list[i]["edge_index"].shape, dtype=torch.long)
        ss_edge_index.append(edge_index)
        index += ss_info_list[i]["node_feats"].shape[0]
        #print(index)
    ss_edge_index = torch.cat(ss_edge_index, dim=1)


    if ss_node_feats.shape[0] !=  S.shape[0]:
          print("ERROR 数据不匹配")
    #print(S.device)
    return ss_node_feats.to(S.device), ss_edge_feats.to(S.device), ss_edge_index.to(S.device)


def save_RNA_sequence(sequence, name, output_dir='./test/output/'):
    """保存RNA序列到FASTA文件"""
    print(name,':',sequence)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{name}.fasta")
    with open(file_path, 'w') as f:
        f.write(f">{name}\n")
        f.write(f"{sequence}\n")
