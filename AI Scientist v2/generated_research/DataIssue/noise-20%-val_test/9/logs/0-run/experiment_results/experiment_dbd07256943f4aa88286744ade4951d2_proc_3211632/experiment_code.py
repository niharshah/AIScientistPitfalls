# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, time, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------- #
#                              HOUSE-KEEPING                                 #
# -------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------- #
#                             HYPER-PARAMETERS                               #
# -------------------------------------------------------------------------- #
BATCH_SIZE = 256
VAL_BATCH = 512
EPOCHS = 10
LR = 3e-3
WEIGHT_DECAY = 1e-4
L1_LAMBDA = 2e-3  # sparsity on rule head
GATE_LAMBDA = 1e-3  # encourage confident gate (towards 0 or 1)
EMB_DIM = 48
CONV_CH = 96
KERNELS = [3, 4, 5]
MAX_LEN = 128
PAD_IDX = 0


# -------------------------------------------------------------------------- #
#                              LOAD DATASET                                  #
# -------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Dataset loaded:", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------------- #
#                               VOCAB                                        #
# -------------------------------------------------------------------------- #
chars = set(itertools.chain.from_iterable(spr["train"]["sequence"]))
char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 reserved for PAD
idx2char = {i: c for c, i in char2idx.items()}
VOCAB = len(char2idx) + 1


def seq_to_bag(seq: str) -> np.ndarray:
    v = np.zeros(VOCAB - 1, dtype=np.float32)
    for ch in seq[:MAX_LEN]:
        v[char2idx[ch] - 1] += 1
    if len(seq) > 0:
        v /= len(seq)
    return v


def seq_to_idx(seq: str) -> np.ndarray:
    arr = np.full(MAX_LEN, PAD_IDX, dtype=np.int64)
    for i, ch in enumerate(seq[:MAX_LEN]):
        arr[i] = char2idx[ch]
    return arr


class SPRDataset(Dataset):
    def __init__(self, split):
        self.bags = np.stack([seq_to_bag(s) for s in split["sequence"]])
        self.seqs = np.stack([seq_to_idx(s) for s in split["sequence"]])
        self.labels = np.array(split["label"], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.bags[idx]),
            torch.from_numpy(self.seqs[idx]),
            torch.tensor(self.labels[idx]),
        )


train_ds, val_ds, test_ds = [SPRDataset(spr[s]) for s in ("train", "dev", "test")]
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH)
test_loader = DataLoader(test_ds, batch_size=VAL_BATCH)

NUM_CLASSES = int(
    max(train_ds.labels.max(), val_ds.labels.max(), test_ds.labels.max()) + 1
)
print("Vocab", VOCAB - 1, "classes", NUM_CLASSES)


# -------------------------------------------------------------------------- #
#                               MODEL                                        #
# -------------------------------------------------------------------------- #
class GatedHybridRuleCNN(nn.Module):
    def __init__(self, vocab: int, n_cls: int):
        super().__init__()
        self.rule_head = nn.Linear(vocab - 1, n_cls)  # interpretable rules
        self.gate = nn.Sequential(nn.Linear(vocab - 1, 1), nn.Sigmoid())  # scalar gate
        self.embed = nn.Embedding(vocab, EMB_DIM, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, CONV_CH, k) for k in KERNELS])
        self.cnn_head = nn.Linear(CONV_CH * len(KERNELS), n_cls)

    def forward(self, bag, seq):
        rule_logits = self.rule_head(bag)  # (B,C)
        gate = self.gate(bag)  # (B,1)
        x = self.embed(seq).transpose(1, 2)  # (B,E,L)
        features = [torch.amax(torch.relu(conv(x)), dim=2) for conv in self.convs]
        cnn_logits = self.cnn_head(torch.cat(features, dim=1))
        final_logits = gate * rule_logits + (1 - gate) * cnn_logits
        return final_logits, rule_logits, gate.squeeze(1)


model = GatedHybridRuleCNN(VOCAB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------------------------------------------------------------- #
#                           METRIC STORAGE                                   #
# -------------------------------------------------------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------------------------------------------------------------------- #
#                               EVALUATION                                   #
# -------------------------------------------------------------------------- #
def evaluate(loader):
    model.eval()
    tot, corr, loss_sum, fid_tot, fid_match = 0, 0, 0.0, 0, 0
    with torch.no_grad():
        for bag, seq, y in loader:
            bag, seq, y = bag.to(device), seq.to(device), y.to(device)
            logits, rule_logits, _ = model(bag, seq)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            rule_preds = rule_logits.argmax(1)
            corr += (preds == y).sum().item()
            fid_match += (preds == rule_preds).sum().item()
            fid_tot += preds.size(0)
            loss_sum += loss.item() * y.size(0)
            tot += y.size(0)
    return corr / tot, loss_sum / tot, fid_match / fid_tot


# -------------------------------------------------------------------------- #
#                               TRAIN LOOP                                   #
# -------------------------------------------------------------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_corr, n_seen = 0.0, 0, 0
    t0 = time.time()
    for bag, seq, y in train_loader:
        bag, seq, y = bag.to(device), seq.to(device), y.to(device)
        optimizer.zero_grad()
        logits, rule_logits, gate = model(bag, seq)
        ce_loss = criterion(logits, y)
        l1_loss = model.rule_head.weight.abs().mean()
        gate_reg = torch.mean(gate * (1 - gate))  # small when gate ~0/1
        loss = ce_loss + L1_LAMBDA * l1_loss + GATE_LAMBDA * gate_reg
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * y.size(0)
        run_corr += (logits.argmax(1) == y).sum().item()
        n_seen += y.size(0)

    train_loss, train_acc = run_loss / n_seen, run_corr / n_seen
    val_acc, val_loss, val_rfs = evaluate(val_loader)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["Rule_Fidelity"].append(val_rfs)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | RFS={val_rfs:.3f}"
    )

# -------------------------------------------------------------------------- #
#                               FINAL TEST                                   #
# -------------------------------------------------------------------------- #
test_acc, test_loss, test_rfs = evaluate(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}, RuleFidelity={test_rfs:.3f}")

# store detailed predictions
model.eval()
preds, gts = [], []
with torch.no_grad():
    for bag, seq, y in test_loader:
        bag, seq = bag.to(device), seq.to(device)
        logits, _, _ = model(bag, seq)
        preds.append(logits.argmax(1).cpu())
        gts.append(y)
experiment_data["SPR_BENCH"]["predictions"] = torch.cat(preds).numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = torch.cat(gts).numpy()

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", working_dir)
