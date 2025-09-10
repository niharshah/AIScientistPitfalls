import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory boilerplate ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(v if t == p else 0 for v, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- vocab ----------
class Vocab:
    def __init__(self, tokens):
        self.itos = ["<pad>", "<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]

    def __len__(self):
        return len(self.itos)


# ---------- model ----------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, feat_dim, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + feat_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, tok_mat, mask, feats):
        emb = self.embedding(tok_mat)  # [B, L, D]
        emb = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)  # mean pool
        x = torch.cat([emb, feats], dim=-1)
        return self.fc(x)


# ---------- load data ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def seq_to_tensor(seq, max_len):
    toks = vocab.encode(seq.split()[:max_len])
    pad_len = max_len - len(toks)
    return toks + [0] * pad_len, [1] * len(toks) + [0] * pad_len


MAX_LEN = 40  # most sequences are short; truncate if necessary


def collate(batch):
    tok_batch, mask_batch, feat_batch, lab_batch = [], [], [], []
    for ex in batch:
        tok_ids, mask = seq_to_tensor(ex["sequence"], MAX_LEN)
        sv = count_shape_variety(ex["sequence"])
        cv = count_color_variety(ex["sequence"])
        ln = len(ex["sequence"].split())
        feat_batch.append([sv, cv, ln, sv / (ln + 1e-6), cv / (ln + 1e-6)])
        tok_batch.append(tok_ids)
        mask_batch.append(mask)
        lab_batch.append(label2id[ex["label"]])
    return (
        torch.tensor(tok_batch).to(device),
        torch.tensor(mask_batch, dtype=torch.float32).to(device),
        torch.tensor(feat_batch, dtype=torch.float32).to(device),
        torch.tensor(lab_batch).to(device),
    )


batch_size = 256
train_loader = DataLoader(
    spr["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    spr["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    spr["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# focal loss (optional)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


model = HybridClassifier(len(vocab), 64, 5, len(labels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = FocalLoss(gamma=1.5)

# ---------- training loop ----------
best_val_swa, patience, epochs = 0.0, 3, 15
no_improve = 0
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for tok, mask, feat, lab in train_loader:
        optimizer.zero_grad()
        out = model(tok, mask, feat)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * lab.size(0)
    train_loss = running_loss / len(spr["train"])

    # ---- validation
    model.eval()
    val_loss, y_true, y_pred, seqs = 0.0, [], [], []
    with torch.no_grad():
        for (tok, mask, feat, lab), idx in zip(dev_loader, range(len(dev_loader))):
            out = model(tok, mask, feat)
            val_loss += nn.functional.cross_entropy(out, lab, reduction="sum").item()
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in lab.cpu().tolist()])
            seqs.extend(
                spr["dev"]["sequence"][
                    idx * batch_size : idx * batch_size + lab.size(0)
                ]
            )
    val_loss /= len(spr["dev"])
    val_swa = shape_weighted_accuracy(seqs, y_true, y_pred)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={val_swa:.4f}"
    )

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test evaluation ----------
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
y_true, y_pred, seqs = [], [], []
with torch.no_grad():
    for (tok, mask, feat, lab), idx in zip(test_loader, range(len(test_loader))):
        out = model(tok, mask, feat)
        preds = out.argmax(1).cpu().tolist()
        y_pred.extend([id2label[p] for p in preds])
        y_true.extend([id2label[i] for i in lab.cpu().tolist()])
        seqs.extend(
            spr["test"]["sequence"][idx * batch_size : idx * batch_size + lab.size(0)]
        )
test_swa = shape_weighted_accuracy(seqs, y_true, y_pred)
print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
