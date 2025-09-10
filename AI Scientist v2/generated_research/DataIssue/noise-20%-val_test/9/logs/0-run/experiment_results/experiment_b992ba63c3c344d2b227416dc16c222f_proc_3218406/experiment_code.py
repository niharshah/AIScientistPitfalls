# Rule-Free CNN Baseline (No Rule Head, No Gate)
import os, time, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -------------------- House-keeping -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- Hyper-params -------------------- #
BATCH_SIZE, VAL_BATCH, EPOCHS = 256, 512, 10
LR, WEIGHT_DECAY = 3e-3, 1e-4
EMB_DIM, CONV_CH, MAX_LEN, PAD_IDX = 48, 96, 128, 0
KERNELS = [3, 4, 5]


# -------------------- Dataset -------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ("train", "dev", "test")})


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# -------------------- Vocab / helpers -------------------- #
chars = set(itertools.chain.from_iterable(spr["train"]["sequence"]))
char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 for PAD
VOCAB = len(char2idx) + 1


def seq_to_bag(seq: str) -> np.ndarray:
    v = np.zeros(VOCAB - 1, np.float32)
    for ch in seq[:MAX_LEN]:
        v[char2idx[ch] - 1] += 1
    if len(seq) > 0:
        v /= len(seq)
    return v


def seq_to_idx(seq: str) -> np.ndarray:
    arr = np.full(MAX_LEN, PAD_IDX, np.int64)
    for i, ch in enumerate(seq[:MAX_LEN]):
        arr[i] = char2idx[ch]
    return arr


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
print("Vocab:", VOCAB - 1, "Classes:", NUM_CLASSES)


# -------------------- Model -------------------- #
class PureCNN(nn.Module):
    """No rule head, no gate – just CNN encoder + linear head"""

    def __init__(self, vocab: int, n_cls: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, EMB_DIM, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, CONV_CH, k) for k in KERNELS])
        self.cnn_head = nn.Linear(CONV_CH * len(KERNELS), n_cls)

    def forward(self, seq):
        x = self.embed(seq).transpose(1, 2)  # (B,E,L)
        feats = [torch.amax(torch.relu(cv(x)), dim=2) for cv in self.convs]
        return self.cnn_head(torch.cat(feats, 1))  # (B,C)


model = PureCNN(VOCAB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------- Experiment data container -------------------- #
experiment_data = {
    "RuleFreeCNN": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -------------------- Evaluation -------------------- #
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for _, seq, y in loader:
        seq, y = seq.to(device), y.to(device)
        logits = model(seq)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total  # acc, loss


# -------------------- Training loop -------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, tr_corr, seen = 0.0, 0, 0
    for _, seq, y in train_loader:
        seq, y = seq.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * y.size(0)
        tr_corr += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    train_acc = tr_corr / seen
    train_loss = tr_loss / seen
    val_acc, val_loss = evaluate(val_loader)

    ed = experiment_data["RuleFreeCNN"]["SPR_BENCH"]
    ed["metrics"]["train"].append(train_acc)
    ed["metrics"]["val"].append(val_acc)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
    )

# -------------------- Final test -------------------- #
test_acc, test_loss = evaluate(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}")

# store predictions / gts
model.eval()
preds, gts = [], []
with torch.no_grad():
    for _, seq, y in test_loader:
        seq = seq.to(device)
        preds.append(model(seq).argmax(1).cpu())
        gts.append(y)
ed = experiment_data["RuleFreeCNN"]["SPR_BENCH"]
ed["predictions"] = torch.cat(preds).numpy()
ed["ground_truth"] = torch.cat(gts).numpy()

# -------------------- Save -------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", working_dir)
