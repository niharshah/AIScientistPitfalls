import os, pathlib, math, time, json, random, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# --------------------------- house-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- metric helpers --------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# --------------------------- data loading ----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# --------------------------- vocab & encoding ------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(dataset):
    tokens = set()
    for seq in dataset["sequence"]:
        tokens.update(seq.strip().split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for t in sorted(tokens):
        vocab[t] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")


def encode_sequence(seq, vocab=vocab):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Labels: {label_set}")


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "seq_enc": torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    seqs = [b["seq_enc"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=vocab[PAD_TOKEN]
    )
    return {"input_ids": padded, "labels": labels, "raw_seq": raw}


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
test_ds = SPRTorchDataset(spr["test"])


# data loaders will be recreated per run because of different batch sizes if needed
def make_loaders():
    return (
        DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn),
        DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn),
        DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn),
    )


# --------------------------- model -----------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, num_labels=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h = self.gru(x)  # h: (num_layers, B, H)
        logits = self.fc(h[-1])  # last layer hidden state
        return logits


# --------------------------- evaluation ------------------------------
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inp)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    acc = correct / total
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    novel_mask = [rule_signature(s) not in train_signatures for s in all_seq]
    novel_total = sum(novel_mask)
    novel_correct = sum(
        int(p == t) for p, t, n in zip(all_pred, all_true, novel_mask) if n
    )
    nrgs = novel_correct / novel_total if novel_total else 0.0
    return loss_sum / total, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# --------------------------- experiment log --------------------------
experiment_data = {
    "num_gru_layers": {
        # filled below per layer
    }
}

# --------------------------- hyper-parameter sweep -------------------
NUM_LAYERS_CANDIDATES = [1, 2, 3]
EPOCHS = 5
best_dev_acc, best_state_dict, best_layer = -1, None, None
best_metrics_test = None
best_preds, best_trues, best_seqs = [], [], []

for n_layers in NUM_LAYERS_CANDIDATES:
    print(f"\n=== Training model with {n_layers} GRU layer(s) ===")
    train_loader, dev_loader, test_loader = make_loaders()
    model = GRUClassifier(
        len(vocab), emb=32, hidden=64, num_labels=len(label_set), num_layers=n_layers
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage for this setting
    exp_key = f"layers_{n_layers}"
    experiment_data["num_gru_layers"][exp_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
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

        val_loss, val_acc, val_swa, val_cwa, val_nrgs, *_ = evaluate(
            model, dev_loader, criterion
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} "
            f"SWA={val_swa:.3f} CWA={val_cwa:.3f} NRGS={val_nrgs:.3f}"
        )

        experiment_data["num_gru_layers"][exp_key]["losses"]["train"].append(train_loss)
        experiment_data["num_gru_layers"][exp_key]["losses"]["val"].append(val_loss)
        experiment_data["num_gru_layers"][exp_key]["metrics"]["train"].append(
            {"epoch": epoch}
        )
        experiment_data["num_gru_layers"][exp_key]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )

    # evaluate on test set
    test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues, seqs = evaluate(
        model, test_loader, criterion
    )
    print(
        f"TEST ({n_layers} layers)  loss={test_loss:.4f}  acc={test_acc:.3f}  "
        f"SWA={test_swa:.3f} CWA={test_cwa:.3f} NRGS={test_nrgs:.3f}"
    )

    experiment_data["num_gru_layers"][exp_key]["predictions"] = preds
    experiment_data["num_gru_layers"][exp_key]["ground_truth"] = trues
    experiment_data["num_gru_layers"][exp_key]["metrics"]["test"] = {
        "loss": test_loss,
        "acc": test_acc,
        "swa": test_swa,
        "cwa": test_cwa,
        "nrgs": test_nrgs,
    }

    # keep best on dev accuracy
    final_val_acc = experiment_data["num_gru_layers"][exp_key]["metrics"]["val"][-1][
        "acc"
    ]
    if final_val_acc > best_dev_acc:
        best_dev_acc = final_val_acc
        best_state_dict = model.state_dict()
        best_layer = n_layers
        best_metrics_test = (test_loss, test_acc, test_swa, test_cwa, test_nrgs)
        best_preds, best_trues, best_seqs = preds, trues, seqs

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nBest model had {best_layer} layer(s) with dev_acc={best_dev_acc:.3f}")
print(
    f"Best TEST results: loss={best_metrics_test[0]:.4f}  "
    f"acc={best_metrics_test[1]:.3f}  "
    f"SWA={best_metrics_test[2]:.3f}  CWA={best_metrics_test[3]:.3f}  "
    f"NRGS={best_metrics_test[4]:.3f}"
)

# quick visualisation for best model
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Acc", "SWA", "CWA", "NRGS"], best_metrics_test[1:], color="skyblue")
ax.set_ylim(0, 1)
ax.set_title(f"Best ({best_layer}-layer) SPR_BENCH Test Metrics")
plt.tight_layout()
plot_path = os.path.join(working_dir, "spr_metrics_bar.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
