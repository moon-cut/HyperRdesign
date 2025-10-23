import torch
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.config import *
from src.function import *
from src.model import *

config = Config()
data_config = config.data_config
train_config = config.train_config
seeding(config.seed)
test_dataset = RNADataset_without_ss_train(
    npy_dir=data_config.test_npy_data_dir,
)

test_loader = DataLoader(test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=featurize)

model = RNAModel_without_ss(config.model_config).to(config.device)
#print(model)
checkpoint_path = train_config.ckpt_path_without_ss
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'), strict=True)   
alphabet = 'AUCG'
t_start = time.time()
model.eval()
with torch.no_grad():
    recovery_list = []
    for batch in test_loader:
        X, S, mask, lengths, names = batch
        X = X.to(config.device)
        S = S.to(config.device)
        mask = mask.to(config.device)
        logits, S = model(X, S, mask)
        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)
        start_idx = 0
        for i,length in enumerate(lengths):
            end_idx = start_idx + length.item()
            sample = samples[start_idx: end_idx]
            RNA_seq = ''.join(alphabet[i] for i in sample)
            #print(RNA_seq)
            save_RNA_sequence(RNA_seq, names[i], output_dir=data_config.test_output_data_dir)
            start_idx = end_idx

#torch.save(model.state_dict(), 'RDesign_modle.pt')
t_end = time.time()
h, m, s = format_time_clean(t_end - t_start)
print(f"推理运行时间: {h:02d}:{m:02d}:{s:05.2f}")
