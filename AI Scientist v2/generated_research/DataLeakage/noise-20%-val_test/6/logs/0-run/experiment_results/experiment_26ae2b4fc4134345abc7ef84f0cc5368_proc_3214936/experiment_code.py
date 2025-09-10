import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------  WORK DIR  ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------  DEVICE  -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------  REPRODUCIBILITY ------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------  LOAD DATA --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})

# -----------------  VOCAB   -----------------------------
all_chars = set("".join(dsets["train"]["sequence"]))
char2id = {c: i + 1 for i, c in enumerate(sorted(all_chars))}  # 0 = PAD
vocab_size = len(char2id) + 1
print(f"Vocab size (incl PAD): {vocab_size}")

labels = sorted(list(set(dsets["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Classes: {labels}")


def seq_to_ids(seq):
    return [char2id[c] for c in seq]


def preprocess_split(split):
    seq_ids = [seq_to_ids(s) for s in dsets[split]["sequence"]]
    labs = [label2id[l] for l in dsets[split]["label"]]
    return seq_ids, labs


train_ids, train_lab = preprocess_split("train")
dev_ids, dev_lab = preprocess_split("dev")
test_ids, test_lab = preprocess_split("test")


# ----------------  DATASET ------------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"seq_ids": self.seqs[idx], "label": self.labels[idx]}


def collate(batch):
    lens = [len(b["seq_ids"]) for b in batch]
    max_len = max(lens)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : len(b["seq_ids"])] = torch.tensor(b["seq_ids"], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"x": padded.to(device), "y": labels.to(device)}


bs = 64
train_loader = DataLoader(
    SeqDataset(train_ids, train_lab), batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(SeqDataset(dev_ids, dev_lab), batch_size=bs, collate_fn=collate)
test_loader = DataLoader(
    SeqDataset(test_ids, test_lab), batch_size=bs, collate_fn=collate
)


# -----------------  MODEL  ------------------------------
class CharCNN(nn.Module):
    def __init__(
        self, vocab, embed_dim, num_classes, n_filters=64, k_sizes=(2, 3, 4, 5)
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k) for k in k_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(n_filters * len(k_sizes), num_classes)

    def extract_features(self, x):
        emb = self.embedding(x).transpose(1, 2)  # B x E x L
        feats = [torch.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        feat = torch.cat(feats, dim=1)  # B x F
        return self.dropout(feat)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.linear(feat)


model = CharCNN(vocab_size, 16, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --------------  METRIC / STORAGE -----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rfa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": test_lab,
    }
}


def accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        preds = logits.argmax(1)
        tot += len(preds)
        correct += (preds == batch["y"]).sum().item()
        loss_sum += loss.item() * len(preds)
    return correct / tot, loss_sum / tot


@torch.no_grad()
def rule_fidelity(model, loader, top_k=50):
    W = model.linear.weight.detach()
    absW = W.abs()
    _, idx = torch.topk(absW, top_k, dim=1)
    mask = torch.zeros_like(W).scatter_(1, idx, 1.0)
    W_trunc = W * mask
    b = model.linear.bias.detach()
    model.eval()
    matches, total = 0, 0
    for batch in loader:
        feats = model.extract_features(batch["x"])
        full = torch.matmul(feats, W.t()) + b
        trunc = torch.matmul(feats, W_trunc.t()) + b
        matches += (full.argmax(1) == trunc.argmax(1)).sum().item()
        total += feats.size(0)
    return matches / total


# -------------------- TRAIN -----------------------------
epochs = 10
best_val = -1
for ep in range(1, epochs + 1):
    # ---- train ----
    model.train()
    seen, correct, loss_sum = 0, 0, 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        bs_ = len(batch["y"])
        seen += bs_
        loss_sum += loss.item() * bs_
        correct += (logits.argmax(1) == batch["y"]).sum().item()
    train_acc = correct / seen
    train_loss = loss_sum / seen

    # ---- dev ----
    val_acc, val_loss = evaluate(model, dev_loader)
    rfa = rule_fidelity(model, dev_loader, top_k=50)

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rfa"].append(rfa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} RFA={rfa:.3f}"
    )

    # save best
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))

# ------------------- TEST EVAL --------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
test_acc, test_loss = evaluate(model, test_loader)
print(f"Best dev_acc={best_val:.3f} --> test_acc={test_acc:.3f}")

# store predictions
model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch["x"])
        preds.extend(logits.argmax(1).cpu().tolist())
experiment_data["SPR_BENCH"]["predictions"] = np.array(preds)

# convert lists to np arrays for saving
for k in ["train_acc", "val_acc", "rfa"]:
    experiment_data["SPR_BENCH"]["metrics"][k] = np.array(
        experiment_data["SPR_BENCH"]["metrics"][k]
    )
for k in ["train", "val"]:
    experiment_data["SPR_BENCH"]["losses"][k] = np.array(
        experiment_data["SPR_BENCH"]["losses"][k]
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics & predictions to working/experiment_data.npy")
