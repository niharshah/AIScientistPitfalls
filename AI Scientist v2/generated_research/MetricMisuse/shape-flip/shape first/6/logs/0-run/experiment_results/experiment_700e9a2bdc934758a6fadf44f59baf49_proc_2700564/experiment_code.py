import os, pathlib, time, json, random, itertools, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# --------------------------- house-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- metrics ---------------------------------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.strip().split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.strip().split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(ww if t == p else 0 for ww, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(ww if t == p else 0 for ww, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def rule_signature(seq):
    return " ".join(tok[0] for tok in seq.strip().split() if tok)


# --------------------------- data ------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(ds):
    toks = set(
        itertools.chain.from_iterable(seq.strip().split() for seq in ds["sequence"])
    )
    vocab = {PAD: 0, UNK: 1}
    for t in sorted(toks):
        vocab[t] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab:", len(vocab))


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


class SPRTorchDS(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2idx[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "seq_enc": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    seqs = [b["seq_enc"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=vocab[PAD])
    return {"input_ids": padded, "labels": labels, "raw_seq": raw}


train_ds, dev_ds, test_ds = (
    SPRTorchDS(spr["train"]),
    SPRTorchDS(spr["dev"]),
    SPRTorchDS(spr["test"]),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)


# --------------------------- model -----------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, num_labels=2, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        _, h = self.gru(x)
        h = self.dropout(h.squeeze(0))
        return self.fc(h)


# --------------------------- experiment log --------------------------
experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


def evaluate(model, loader, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inp)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * len(labels)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            tot += len(labels)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    acc = correct / tot
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    novel = [rule_signature(s) not in train_signatures for s in all_seq]
    n_tot = sum(novel)
    n_cor = sum(int(p == t) for p, t, n in zip(all_pred, all_true, novel) if n)
    nrgs = n_cor / n_tot if n_tot else 0.0
    return loss_sum / tot, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# --------------------------- tuning loop -----------------------------
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS = 5
val_acc_end = {}
for dr in dropout_rates:
    tag = f"{dr:.1f}"
    print(f"\n=== Training with dropout={dr} ===")
    model = GRUClassifier(
        len(vocab), emb=32, hidden=64, num_labels=len(label_set), dropout=dr
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
        train_loss = epoch_loss / len(train_ds)
        val_loss, val_acc, val_swa, val_cwa, val_nrgs, _, _, _ = evaluate(
            model, dev_loader, criterion
        )
        val_acc_end[dr] = val_acc
        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f} swa={val_swa:.3f} cwa={val_cwa:.3f} nrgs={val_nrgs:.3f}"
        )
        log["losses"]["train"].append(train_loss)
        log["losses"]["val"].append(val_loss)
        log["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )
        log["timestamps"].append(time.time())
    # final test
    t_loss, t_acc, t_swa, t_cwa, t_nrgs, preds, trues, seqs = evaluate(
        model, test_loader, criterion
    )
    log["metrics"]["test"] = {
        "loss": t_loss,
        "acc": t_acc,
        "swa": t_swa,
        "cwa": t_cwa,
        "nrgs": t_nrgs,
    }
    log["predictions"] = preds
    log["ground_truth"] = trues
    experiment_data["dropout_rate"]["SPR_BENCH"][tag] = log
    print(f"Test | acc={t_acc:.3f} swa={t_swa:.3f} cwa={t_cwa:.3f} nrgs={t_nrgs:.3f}")

# --------------------------- visualisation ---------------------------
best_dr = max(val_acc_end, key=val_acc_end.get)
best_metrics = experiment_data["dropout_rate"]["SPR_BENCH"][f"{best_dr:.1f}"][
    "metrics"
]["test"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    ["Acc", "SWA", "CWA", "NRGS"],
    [
        best_metrics["acc"],
        best_metrics["swa"],
        best_metrics["cwa"],
        best_metrics["nrgs"],
    ],
    color="skyblue",
)
ax.set_ylim(0, 1)
ax.set_title(f"Best Dropout={best_dr:.1f} Test Metrics")
plt.tight_layout()
plot_path = os.path.join(working_dir, "spr_metrics_bar.png")
plt.savefig(plot_path)
print(f"\nPlot saved to {plot_path}")

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
