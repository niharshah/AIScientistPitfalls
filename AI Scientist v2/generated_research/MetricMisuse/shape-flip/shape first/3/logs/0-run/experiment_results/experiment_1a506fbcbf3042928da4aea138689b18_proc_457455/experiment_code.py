import os, pathlib, numpy as np, torch
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------------- basic setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- utilities ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


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
    return 0.0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


class Vocab:
    def __init__(self, toks: List[str]):
        self.itos = ["<pad>"] + sorted(set(toks))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, toks: List[str]):
        return [self.stoi[t] for t in toks]


class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_cls: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, n_cls)

    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))


# ---------------- data path ------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"SPR_BENCH not found at {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------- vocab / labels -------------
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def collate_batch(batch):
    token_ids, offsets, lbls = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        lbls.append(label2id[ex["label"]])
    text = torch.tensor(token_ids, dtype=torch.long).to(device)
    offs = torch.tensor(offsets[:-1], dtype=torch.long).to(device)
    labels_t = torch.tensor(lbls, dtype=torch.long).to(device)
    return text, offs, labels_t


# ---------------- experiment dict ------------
experiment_data = {"batch_size_tuning": {}}

# ---------------- evaluation helper ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    y_t, y_p, seqs, tot_loss = [], [], [], 0.0
    idx = 0
    with torch.no_grad():
        for text, offs, lbl in loader:
            out = model(text, offs)
            loss = criterion(out, lbl)
            tot_loss += loss.item() * lbl.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_p.extend([id2label[p] for p in preds])
            y_t.extend([id2label[i] for i in lbl.cpu().tolist()])
            seqs.extend(loader.dataset["sequence"][idx : idx + lbl.size(0)])
            idx += lbl.size(0)
    avg_loss = tot_loss / len(y_t)
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    hwa = harmonic_weighted_accuracy(seqs, y_t, y_p)
    return avg_loss, swa, cwa, hwa, y_t, y_p


# ---------------- hyper-parameter sweep -------
batch_sizes = [32, 64, 128, 256]
epochs = 5
embed_dim = 64
for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    # dataloaders
    train_loader = DataLoader(
        spr["train"], batch_size=bs, shuffle=True, collate_fn=collate_batch
    )
    dev_loader = DataLoader(
        spr["dev"], batch_size=bs, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        spr["test"], batch_size=bs, shuffle=False, collate_fn=collate_batch
    )

    # model / optim
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage for this run
    run_key = f"bs_{bs}"
    experiment_data["batch_size_tuning"][run_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # training loop
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for text, offs, lbl in train_loader:
            optimizer.zero_grad()
            out = model(text, offs)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * lbl.size(0)
        train_loss = run_loss / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)

        experiment_data["batch_size_tuning"][run_key]["losses"]["train"].append(
            train_loss
        )
        experiment_data["batch_size_tuning"][run_key]["losses"]["val"].append(val_loss)
        experiment_data["batch_size_tuning"][run_key]["metrics"]["train"].append(None)
        experiment_data["batch_size_tuning"][run_key]["metrics"]["val"].append(
            {"SWA": swa, "CWA": cwa, "HWA": hwa}
        )
        print(
            f"Epoch {ep}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | HWA {hwa:.4f}"
        )

    # final test evaluation
    test_loss, swa_t, cwa_t, hwa_t, y_t, y_p = evaluate(model, test_loader)
    experiment_data["batch_size_tuning"][run_key]["predictions"] = y_p
    experiment_data["batch_size_tuning"][run_key]["ground_truth"] = y_t
    experiment_data["batch_size_tuning"][run_key]["test_metrics"] = {
        "loss": test_loss,
        "SWA": swa_t,
        "CWA": cwa_t,
        "HWA": hwa_t,
    }
    print(f"Test | loss {test_loss:.4f} | HWA {hwa_t:.4f}")

# ---------------- save -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to working/experiment_data.npy")
