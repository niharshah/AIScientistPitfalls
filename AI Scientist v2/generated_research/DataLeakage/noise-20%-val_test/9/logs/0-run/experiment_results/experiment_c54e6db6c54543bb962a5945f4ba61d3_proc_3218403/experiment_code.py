import os, time, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ------------------------------------------------- SET-UP ------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------- HYPER-PARAMETERS ------------------------------------------- #
BATCH_SIZE, VAL_BATCH, EPOCHS = 256, 512, 10
LR, WEIGHT_DECAY, L1_LAMBDA, GATE_LAMBDA = 3e-3, 1e-4, 2e-3, 1e-3
EMB_DIM, CONV_CH, KERNELS = 48, 96, [3, 4, 5]
MAX_LEN, PAD_IDX = 128, 0


# ---------------------------------------------- DATA LOADING --------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {split: _load(f"{split}.csv") for split in ("train", "dev", "test")}
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------- VOCAB ------------------------------------------------- #
chars = set(itertools.chain.from_iterable(spr["train"]["sequence"]))
char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 reserved for PAD
idx2char = {i: c for c, i in char2idx.items()}
VOCAB = len(char2idx) + 1


def seq_to_bag(seq: str) -> np.ndarray:
    v = np.zeros(VOCAB - 1, dtype=np.float32)
    for ch in seq[:MAX_LEN]:
        v[char2idx[ch] - 1] += 1
    if len(seq):
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
        self.label = np.array(split["label"], dtype=np.int64)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.bags[idx]),
            torch.from_numpy(self.seqs[idx]),
            torch.tensor(self.label[idx]),
        )


train_ds, val_ds, test_ds = (SPRDataset(spr[s]) for s in ("train", "dev", "test"))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH)
test_loader = DataLoader(test_ds, batch_size=VAL_BATCH)

NUM_CLASSES = int(
    max(train_ds.label.max(), val_ds.label.max(), test_ds.label.max()) + 1
)
print(f"Vocab size = {VOCAB-1}, #Classes = {NUM_CLASSES}")


# -------------------------------------------------- MODEL ------------------------------------------------- #
class CNNFeatureGateHybrid(nn.Module):
    """
    Gate learns a confidence score from CNN features.
    Corrected blend:   logits = g * cnn + (1-g) * rule
    """

    def __init__(self, vocab: int, n_cls: int):
        super().__init__()
        # Rule branch (bag-of-chars)
        self.rule_head = nn.Linear(vocab - 1, n_cls)
        # CNN branch
        self.embed = nn.Embedding(vocab, EMB_DIM, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(EMB_DIM, CONV_CH, k) for k in KERNELS])
        feat_dim = CONV_CH * len(KERNELS)
        self.cnn_head = nn.Linear(feat_dim, n_cls)
        # Gate
        self.gate = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

    def forward(self, bag, seq):
        rule_logits = self.rule_head(bag.float())  # (B,C)
        x = self.embed(seq).transpose(1, 2)  # (B,E,L)
        pooled = [torch.relu(c(x)).amax(dim=2) for c in self.convs]
        feats = torch.cat(pooled, dim=1)  # (B,feat_dim)
        cnn_logits = self.cnn_head(feats)  # (B,C)
        g = self.gate(feats).squeeze(1)  # (B,)
        logits = g.unsqueeze(1) * cnn_logits + (1 - g.unsqueeze(1)) * rule_logits
        return logits, rule_logits, g


model = CNNFeatureGateHybrid(VOCAB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ------------------------------------------ EXPERIMENT LOGGING ------------------------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------ EVALUATION --------------------------------------------- #
def evaluate(loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    rf_match, rf_tot = 0, 0
    with torch.no_grad():
        for bag, seq, y in loader:
            bag, seq, y = bag.to(device), seq.to(device), y.to(device)
            logits, rule_logits, _ = model(bag, seq)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            rule_preds = rule_logits.argmax(1)
            correct += (preds == y).sum().item()
            rf_match += (preds == rule_preds).sum().item()
            rf_tot += preds.size(0)
            loss_sum += loss.item() * y.size(0)
            tot += y.size(0)
    return correct / tot, loss_sum / tot, rf_match / rf_tot


# ------------------------------------------------- TRAINING ---------------------------------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss, ep_corr, seen = 0.0, 0, 0
    for bag, seq, y in train_loader:
        bag, seq, y = bag.to(device), seq.to(device), y.to(device)
        optimizer.zero_grad()
        logits, rule_logits, gate = model(bag, seq)
        ce = criterion(logits, y)
        l1 = model.rule_head.weight.abs().mean()
        gate_reg = torch.mean(gate * (1 - gate))  # encourage extremes
        loss = ce + L1_LAMBDA * l1 + GATE_LAMBDA * gate_reg
        loss.backward()
        optimizer.step()

        ep_loss += loss.item() * y.size(0)
        ep_corr += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)

    train_loss, train_acc = ep_loss / seen, ep_corr / seen
    val_acc, val_loss, val_rf = evaluate(val_loader)

    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["Rule_Fidelity"].append(val_rf)
    ed["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.3f} | RuleFidelity={val_rf:.3f}"
    )

# -------------------------------------------------- TESTING ---------------------------------------------- #
test_acc, test_loss, test_rf = evaluate(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}, RuleFidelity={test_rf:.3f}")

# store predictions on test set
model.eval()
preds, gts = [], []
with torch.no_grad():
    for bag, seq, y in test_loader:
        bag, seq = bag.to(device), seq.to(device)
        plogits, _, _ = model(bag, seq)
        preds.append(plogits.argmax(1).cpu())
        gts.append(y)
ed = experiment_data["SPR_BENCH"]
ed["predictions"] = torch.cat(preds).numpy()
ed["ground_truth"] = torch.cat(gts).numpy()

# --------------------------------------------- SAVE EVERYTHING ------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", working_dir)
