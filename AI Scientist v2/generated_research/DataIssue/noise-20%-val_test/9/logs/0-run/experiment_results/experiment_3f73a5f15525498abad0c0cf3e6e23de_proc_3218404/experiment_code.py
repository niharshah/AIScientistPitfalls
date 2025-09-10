# No-Rule-Sparsity Ablation (L1_LAMBDA = 0)
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
print("Using:", device)

# -------------------------------------------------------------------------- #
#                             HYPER-PARAMETERS                               #
# -------------------------------------------------------------------------- #
BATCH_SIZE, VAL_BATCH, EPOCHS = 256, 512, 10
LR, WEIGHT_DECAY = 3e-3, 1e-4
L1_LAMBDA = 0.0  # *** ablation: remove sparsity penalty ***
GATE_LAMBDA = 1e-3
EMB_DIM, CONV_CH = 48, 96
KERNELS, MAX_LEN, PAD_IDX = [3, 4, 5], 128, 0


# -------------------------------------------------------------------------- #
#                               LOAD DATASET                                 #
# -------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name: str):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({s: _ld(f"{s}.csv") for s in ("train", "dev", "test")})


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------------- #
#                                 VOCAB                                      #
# -------------------------------------------------------------------------- #
chars = set(itertools.chain.from_iterable(spr["train"]["sequence"]))
char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
idx2char = {i: c for c, i in char2idx.items()}
VOCAB = len(char2idx) + 1


def seq_to_bag(seq: str) -> np.ndarray:
    v = np.zeros(VOCAB - 1, np.float32)
    for ch in seq[:MAX_LEN]:
        v[char2idx[ch] - 1] += 1
    if len(seq):
        v /= len(seq)
    return v


def seq_to_idx(seq: str) -> np.ndarray:
    x = np.full(MAX_LEN, PAD_IDX, np.int64)
    for i, ch in enumerate(seq[:MAX_LEN]):
        x[i] = char2idx[ch]
    return x


class SPRDataset(Dataset):
    def __init__(self, split):
        self.bags = np.stack([seq_to_bag(s) for s in split["sequence"]])
        self.seqs = np.stack([seq_to_idx(s) for s in split["sequence"]])
        self.labels = np.array(split["label"], np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.bags[idx]),
            torch.from_numpy(self.seqs[idx]),
            torch.tensor(self.labels[idx]),
        )


train_ds, val_ds, test_ds = [SPRDataset(spr[s]) for s in ("train", "dev", "test")]
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, VAL_BATCH)
test_loader = DataLoader(test_ds, VAL_BATCH)
NUM_CLASSES = int(
    max(train_ds.labels.max(), val_ds.labels.max(), test_ds.labels.max()) + 1
)
print("vocab", VOCAB - 1, "classes", NUM_CLASSES)


# -------------------------------------------------------------------------- #
#                                  MODEL                                     #
# -------------------------------------------------------------------------- #
class GatedHybridRuleCNN(nn.Module):
    def __init__(self, vocab, n_cls):
        super().__init__()
        self.rule_head = nn.Linear(vocab - 1, n_cls)
        self.gate = nn.Sequential(nn.Linear(vocab - 1, 1), nn.Sigmoid())
        self.embed = nn.Embedding(vocab, EMB_DIM, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, CONV_CH, k) for k in KERNELS])
        self.cnn_head = nn.Linear(CONV_CH * len(KERNELS), n_cls)

    def forward(self, bag, seq):
        rule_logits = self.rule_head(bag)
        gate = self.gate(bag)
        x = self.embed(seq).transpose(1, 2)
        feats = [torch.amax(torch.relu(c(x)), dim=2) for c in self.convs]
        cnn_logits = self.cnn_head(torch.cat(feats, 1))
        return (
            gate * rule_logits + (1 - gate) * cnn_logits,
            rule_logits,
            gate.squeeze(1),
        )


model = GatedHybridRuleCNN(VOCAB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------------------------------------------------------------- #
#                           METRIC STORAGE                                   #
# -------------------------------------------------------------------------- #
experiment_data = {
    "NoRuleSparsity": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
            "losses": {"train": [], "val": []},
            "timestamps": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -------------------------------------------------------------------------- #
#                               UTILITIES                                    #
# -------------------------------------------------------------------------- #
def evaluate(loader):
    model.eval()
    total, correct, lsum, fid_tot, fid_match = 0, 0, 0.0, 0, 0
    with torch.no_grad():
        for bag, seq, y in loader:
            bag, seq, y = bag.to(device), seq.to(device), y.to(device)
            logits, rule_logits, _ = model(bag, seq)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            rule_preds = rule_logits.argmax(1)
            correct += (preds == y).sum().item()
            fid_match += (preds == rule_preds).sum().item()
            fid_tot += preds.size(0)
            lsum += loss.item() * y.size(0)
            total += y.size(0)
    return correct / total, lsum / total, fid_match / fid_tot


# -------------------------------------------------------------------------- #
#                               TRAIN LOOP                                   #
# -------------------------------------------------------------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, tr_corr, seen = 0.0, 0, 0
    for bag, seq, y in train_loader:
        bag, seq, y = bag.to(device), seq.to(device), y.to(device)
        optimizer.zero_grad()
        logits, rule_logits, gate = model(bag, seq)
        ce = criterion(logits, y)
        l1 = model.rule_head.weight.abs().mean()
        gate_reg = torch.mean(gate * (1 - gate))
        loss = ce + L1_LAMBDA * l1 + GATE_LAMBDA * gate_reg  # L1_LAMBDA==0
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * y.size(0)
        tr_corr += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    train_loss, train_acc = tr_loss / seen, tr_corr / seen
    val_acc, val_loss, val_rf = evaluate(val_loader)
    ed = experiment_data["NoRuleSparsity"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["Rule_Fidelity"].append(val_rf)
    ed["timestamps"].append(time.time())
    print(
        f"Ep{epoch}: tr_loss {train_loss:.4f} tr_acc {train_acc:.3f} | "
        f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} RFS {val_rf:.3f}"
    )

# -------------------------------------------------------------------------- #
#                              FINAL TEST                                    #
# -------------------------------------------------------------------------- #
test_acc, test_loss, test_rf = evaluate(test_loader)
print(f"\nTest: loss {test_loss:.4f} acc {test_acc:.3f} RuleFid {test_rf:.3f}")

# store predictions
model.eval()
preds, gts = [], []
with torch.no_grad():
    for bag, seq, y in test_loader:
        p, _, _ = model(bag.to(device), seq.to(device))
        preds.append(p.argmax(1).cpu())
        gts.append(y)
experiment_data["NoRuleSparsity"]["SPR_BENCH"]["predictions"] = torch.cat(preds).numpy()
experiment_data["NoRuleSparsity"]["SPR_BENCH"]["ground_truth"] = torch.cat(gts).numpy()

# save all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
