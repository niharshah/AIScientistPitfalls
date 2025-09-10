import os, pathlib, random, time, math, json, warnings
from typing import List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------------- working dir / device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------- deterministic behaviour -----------------
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------- helper: load SPR_BENCH ---------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# -------------- metrics ---------------------------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.strip().split() if tok})


def shape_weighted_accuracy(seqs: List[str], y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------------- data path -----------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")  # fallback for local tests
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocab & label maps --------------------
def seq_tokens(s):
    return s.strip().split()


vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for tok in seq_tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vsize = len(vocab)
labels = sorted({ex["label"] for ex in spr["train"]})
lab2id = {l: i for i, l in enumerate(labels)}
n_classes = len(labels)
print("Vocab", vsize, "classes", n_classes)


# -------------- torch Dataset ---------------------------
class SPRDataset(Dataset):
    def __init__(self, split, vocab, lab2id):
        self.data = split
        self.vocab = vocab
        self.lab2id = lab2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = [self.vocab.get(t, 1) for t in seq_tokens(row["sequence"])]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.lab2id[row["label"]], dtype=torch.long),
            "raw": row["sequence"],
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    lens = [len(x) for x in ids]
    pad = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw"] for b in batch]
    return {"ids": pad, "lengths": torch.tensor(lens), "label": labels, "raw": raws}


bsz = 256
train_loader = DataLoader(
    SPRDataset(spr["train"], vocab, lab2id),
    batch_size=bsz,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"], vocab, lab2id),
    batch_size=bsz,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRDataset(spr["test"], vocab, lab2id),
    batch_size=bsz,
    shuffle=False,
    collate_fn=collate,
)


# -------------- model -----------------------------------
class NeuroSymbolicClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.sym_ff = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16))
        self.classifier = nn.Linear(d_model + 16, n_classes)

    def forward(self, ids, lengths, sym_feats):
        x = self.emb(ids)
        mask = ids == 0
        x = self.enc(x, src_key_padding_mask=mask)
        avg = (x * (~mask).unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        sym = self.sym_ff(sym_feats)
        concat = torch.cat([avg, sym], dim=1)
        return self.classifier(concat)


# -------------- training setup --------------------------
lr = 1e-3
epochs = 5
d_model = 64
heads = 4
model = NeuroSymbolicClassifier(vsize, d_model, heads, n_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

experiment_data = {
    "spr_bench": {
        "metrics": {"train_acc": [], "dev_acc": [], "dev_swa": []},
        "losses": {"train": [], "dev": []},
        "predictions": {},
        "ground_truth": {
            "dev": [lab2id[l] for l in spr["dev"]["label"]],
            "test": [lab2id[l] for l in spr["test"]["label"]],
        },
    }
}


def make_sym_feats(raws: List[str]):
    feats = np.zeros((len(raws), 3), dtype=np.float32)
    for i, s in enumerate(raws):
        feats[i, 0] = count_shape_variety(s)
        feats[i, 1] = len({tok[1] for tok in s.strip().split() if len(tok) > 1})
        feats[i, 2] = len(seq_tokens(s))
    # simple normalization
    feats[:, 0] /= 10.0
    feats[:, 1] /= 10.0
    feats[:, 2] /= 20.0
    return torch.tensor(feats, dtype=torch.float32)


def evaluate(loader):
    model.eval()
    tot = 0
    correct = 0
    lsum = 0
    preds = []
    raws = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)
            sym = make_sym_feats(batch["raw"]).to(device)
            logits = model(ids, lengths.float(), sym)
            loss = criterion(logits, labels)
            lsum += loss.item() * labels.size(0)
            p = logits.argmax(-1)
            correct += (p == labels).sum().item()
            tot += labels.size(0)
            preds.extend(p.cpu().tolist())
            raws.extend(batch["raw"])
            trues.extend(labels.cpu().tolist())
    acc = correct / tot
    swa = shape_weighted_accuracy(raws, trues, preds)
    return lsum / tot, acc, swa, preds, raws, trues


# -------------- training loop ---------------------------
for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0
    correct = 0
    tot = 0
    for batch in train_loader:
        opt.zero_grad()
        ids = batch["ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["label"].to(device)
        sym = make_sym_feats(batch["raw"]).to(device)
        logits = model(ids, lengths.float(), sym)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        run_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        tot += labels.size(0)
    tr_loss = run_loss / tot
    tr_acc = correct / tot

    dev_loss, dev_acc, dev_swa, dev_pred, _, _ = evaluate(dev_loader)

    experiment_data["spr_bench"]["losses"]["train"].append(tr_loss)
    experiment_data["spr_bench"]["losses"]["dev"].append(dev_loss)
    experiment_data["spr_bench"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["spr_bench"]["metrics"]["dev_acc"].append(dev_acc)
    experiment_data["spr_bench"]["metrics"]["dev_swa"].append(dev_swa)

    print(
        f"Epoch {epoch}: validation_loss = {dev_loss:.4f} | dev_acc={dev_acc:.3f} | dev_SWA={dev_swa:.3f}"
    )

# -------------- final test evaluation -------------------
test_loss, test_acc, test_swa, test_pred, _, _ = evaluate(test_loader)
print(f"TEST: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f}")
experiment_data["spr_bench"]["predictions"]["test"] = test_pred
experiment_data["spr_bench"]["test_metrics"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
}

# -------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot loss curve
plt.figure()
plt.plot(experiment_data["spr_bench"]["losses"]["train"], label="train")
plt.plot(experiment_data["spr_bench"]["losses"]["dev"], label="dev")
plt.legend()
plt.title("CE Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()
