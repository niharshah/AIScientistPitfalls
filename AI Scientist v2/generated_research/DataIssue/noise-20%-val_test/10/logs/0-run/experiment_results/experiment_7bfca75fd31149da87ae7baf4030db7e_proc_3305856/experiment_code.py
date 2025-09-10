import os, pathlib, numpy as np, torch, time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score

# --------------------------  working dir --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------  device -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------  data utils ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not data_root.exists():
    data_root = pathlib.Path("SPR_BENCH/")  # fallback for local tests
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})


# build vocab (characters)
def build_vocab(dataset):
    charset = set()
    for seq in dataset["sequence"]:
        charset.update(seq)
    stoi = {c: i for i, c in enumerate(sorted(charset))}
    return stoi


stoi = build_vocab(spr["train"])
itos = {i: s for s, i in stoi.items()}
V = len(stoi)
num_classes = len(set(spr["train"]["label"]))
print(f"Char vocab size = {V} | Classes = {num_classes}")


# -----------------------  torch dataset ---------------------------
class CharCountDataset(Dataset):
    def __init__(self, hf_split, stoi):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.stoi = stoi
        self.vocab_size = len(stoi)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        vec = torch.zeros(self.vocab_size, dtype=torch.float32)
        for ch in self.seqs[idx]:
            if ch in self.stoi:  # safety
                vec[self.stoi[ch]] += 1.0
        return {"x": vec, "y": torch.tensor(self.labels[idx], dtype=torch.long)}


def collate(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs.to(device), "y": ys.to(device)}


bs = 512
train_dl = DataLoader(
    CharCountDataset(spr["train"], stoi),
    batch_size=bs,
    shuffle=True,
    collate_fn=collate,
)
dev_dl = DataLoader(
    CharCountDataset(spr["dev"], stoi), batch_size=bs, shuffle=False, collate_fn=collate
)
test_dl = DataLoader(
    CharCountDataset(spr["test"], stoi),
    batch_size=bs,
    shuffle=False,
    collate_fn=collate,
)


# -----------------------  model -----------------------------------
class CharLogReg(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        self.lin = nn.Linear(in_dim, num_cls, bias=True)

    def forward(self, x):
        return self.lin(x)


model = CharLogReg(V, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()


# ----------------------- helpers ----------------------------------
def run_epoch(model, loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["y"].size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        gts.extend(batch["y"].detach().cpu().numpy())
    loss_mean = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return loss_mean, f1, preds, gts


# -----------------------  training loop ---------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

epochs = 30
best_val_f1, best_state = 0.0, None
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, train=True)
    val_loss, val_f1, _, _ = run_epoch(model, dev_dl, train=False)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
    if ep % 5 == 0 or ep == 1:
        print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | val_f1 = {val_f1:.4f}")

print(f"Best dev Macro-F1 = {best_val_f1:.4f}")
model.load_state_dict(best_state)

# -----------------------  evaluation ------------------------------
test_loss, test_f1, preds, gts = run_epoch(model, test_dl, train=False)
print(f"Test Macro-F1 = {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# -----------------------  rule extraction -------------------------
with torch.no_grad():
    W = model.lin.weight.detach().cpu().numpy()  # shape [C, V]
top_k = 3
rules = {}
for c in range(num_classes):
    top_idx = W[c].argsort()[-top_k:][::-1]  # highest weights
    rules[c] = top_idx
    human_readable = [itos[i] for i in top_idx]
    print(f"Class {c} rule top-{top_k} chars: {human_readable}")


# rule-based classifier
def rule_predict(seq):
    vec = np.zeros(V, dtype=np.float32)
    for ch in seq:
        if ch in stoi:
            vec[stoi[ch]] += 1.0
    scores = []
    for c in range(num_classes):
        scores.append(vec[rules[c]].sum())  # sum counts of rule chars
    return int(np.argmax(scores))


rule_preds = [rule_predict(s) for s in spr["test"]["sequence"]]
REA = accuracy_score(spr["test"]["label"], rule_preds)
print(f"Rule Extraction Accuracy (REA): {REA:.4f}")

# -----------------------  save data -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
