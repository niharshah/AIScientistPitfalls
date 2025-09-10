import os, pathlib, math, time, json, random, itertools, numpy as np, torch, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

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
    return sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p) / max(
        sum(weights), 1
    )


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    return sum(w for w, t, p in zip(weights, y_true, y_pred) if t == p) / max(
        sum(weights), 1
    )


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# --------------------------- data loading ----------------------------
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

# --------------------------- vocab & encoding ------------------------
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


def build_vocab(dataset):
    tokens = set()
    for seq in dataset["sequence"]:
        tokens.update(seq.strip().split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for t in sorted(tokens):
        vocab[t] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print("Labels:", label_set)


class SPRTorchDataset(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]
        self.labels = [label2idx[l] for l in hf["label"]]

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


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"]),
    SPRTorchDataset(spr["dev"]),
    SPRTorchDataset(spr["test"]),
)
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)


# --------------------------- model -----------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


# --------------------------- evaluation ------------------------------
criterion = nn.CrossEntropyLoss()
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


def evaluate(model, loader):
    model.eval()
    total = correct = loss_sum = 0.0
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
            total += len(labels)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    acc = correct / total
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    novel = [rule_signature(s) not in train_signatures for s in all_seq]
    nrgs = sum(int(p == t) for p, t, n in zip(all_pred, all_true, novel) if n) / max(
        sum(novel), 1
    )
    return loss_sum / total, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# --------------------------- hyper-parameter sweep -------------------
LEARNING_RATES = [5e-4, 1e-4, 5e-5]
EPOCHS = 5
experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

for lr in LEARNING_RATES:
    print(f"\n==========  Training with lr={lr}  ==========")
    model = GRUClassifier(len(vocab), emb=32, hidden=64, num_labels=len(label_set)).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader(128):
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
            model, dev_loader
        )
        print(
            f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"Acc={val_acc:.3f}  SWA={val_swa:.3f}  CWA={val_cwa:.3f}  NRGS={val_nrgs:.3f}"
        )
        run_dict["losses"]["train"].append(train_loss)
        run_dict["losses"]["val"].append(val_loss)
        run_dict["metrics"]["train"].append({"epoch": epoch})
        run_dict["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )
        run_dict["timestamps"].append(time.time())
    # final test
    test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues, seqs = evaluate(
        model, test_loader
    )
    print(
        "TEST  loss={:.4f}  acc={:.3f}  SWA={:.3f} CWA={:.3f} NRGS={:.3f}".format(
            test_loss, test_acc, test_swa, test_cwa, test_nrgs
        )
    )
    run_dict["predictions"] = preds
    run_dict["ground_truth"] = trues
    run_dict["metrics"]["test"] = {
        "loss": test_loss,
        "acc": test_acc,
        "swa": test_swa,
        "cwa": test_cwa,
        "nrgs": test_nrgs,
    }
    # save simple bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Acc", "SWA", "CWA", "NRGS"],
        [test_acc, test_swa, test_cwa, test_nrgs],
        color="skyblue",
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"SPR_BENCH Test Metrics (lr={lr})")
    plt.tight_layout()
    plot_path = os.path.join(working_dir, f"spr_metrics_lr_{lr}.png")
    plt.savefig(plot_path)
    plt.close()
    print("Plot saved to", plot_path)
    experiment_data["learning_rate"]["SPR_BENCH"][f"lr_{lr}"] = run_dict

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiment data saved.")
