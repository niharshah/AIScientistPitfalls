import os, pathlib, numpy as np, torch
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
torch.manual_seed(42)

# ---------- experiment dictionary ----------
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}  # will be filled per‐LR

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- load SPR-BENCH ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")


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


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- metric helpers ----------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(seqs, y_t, y_p):
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    return 0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


# ---------- vocab ----------
class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __call__(self, tokens: List[str]):
        return [self.stoi[t] for t in tokens]

    def __len__(self):
        return len(self.itos)


all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ---------- collate ----------
def collate_batch(batch):
    token_ids, offsets, label_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[ex["label"]])
    return (
        torch.tensor(token_ids, dtype=torch.long).to(device),
        torch.tensor(offsets[:-1], dtype=torch.long).to(device),
        torch.tensor(label_ids, dtype=torch.long).to(device),
    )


batch_size = 128


def make_loader(split, shuffle):
    return DataLoader(
        spr[split], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )


train_loader = make_loader("train", True)
dev_loader = make_loader("dev", False)
test_loader = make_loader("test", False)


# ---------- model ----------
class BagClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, n_cls):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, n_cls)

    def forward(self, txt, offs):
        return self.fc(self.embedding(txt, offs))


# ---------- evaluation ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    y_true, y_pred, seqs, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for b, (txt, offs, lab) in enumerate(loader):
            out = model(txt, offs)
            total_loss += criterion(out, lab).item() * lab.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[i] for i in lab.cpu().tolist()])
            start = b * batch_size
            end = start + lab.size(0)
            seqs.extend(loader.dataset["sequence"][start:end])
    avg_loss = total_loss / len(y_true)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# ---------- training routine ----------
def run_experiment(lr, epochs=5, embed_dim=64):
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_metrics = [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for txt, offs, lab in train_loader:
            opt.zero_grad()
            loss = criterion(model(txt, offs), lab)
            loss.backward()
            opt.step()
            running += loss.item() * lab.size(0)
        tr_loss = running / len(spr["train"])
        vl_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_metrics.append({"SWA": swa, "CWA": cwa, "HWA": hwa})
        print(
            f"LR={lr} | Epoch {ep}: tr_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, HWA={hwa:.4f}"
        )
    # final test
    ts_loss, swa_t, cwa_t, hwa_t, y_t, p_t = evaluate(model, test_loader)
    print(f"LR={lr} | Test: loss={ts_loss:.4f}, HWA={hwa_t:.4f}")
    return {
        "metrics": {"train": [], "val": val_metrics},
        "losses": {"train": train_losses, "val": val_losses},
        "predictions": p_t,
        "ground_truth": y_t,
        "test_metrics": {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t, "loss": ts_loss},
    }


# ---------- grid search ----------
lrs = [5e-4, 1e-3, 2e-3]
for lr in lrs:
    experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = run_experiment(lr)

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
