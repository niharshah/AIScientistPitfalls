import os, pathlib, time, json, random, math, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------------ house-keeping ------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print(f"Using device: {device}")


# ------------------------ metric helpers -----------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# ------------------------ data loading -------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
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

# ------------------------ vocab & encoding ----------------------
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


def encode_sequence(seq, vocab=vocab):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2idx[l] for l in split["label"]]

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


# ------------------------ model ---------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


# ------------------------ evaluation ----------------------------
def evaluate(model, loader, criterion, train_signatures):
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
    novel_mask = [rule_signature(s) not in train_signatures for s in all_seq]
    novel_total = sum(novel_mask)
    novel_correct = sum(
        int(p == t) for p, t, m in zip(all_pred, all_true, novel_mask) if m
    )
    nrgs = novel_correct / novel_total if novel_total else 0.0
    avg_loss = loss_sum / tot
    return avg_loss, acc, swa, cwa, nrgs, all_pred, all_true


# ------------------------ experiment dict -----------------------
experiment_data = {"embedding_dim": {"SPR_BENCH": {"runs": {}}}}

# ------------------------ hyperparam sweep ----------------------
embed_dims = [16, 32, 64, 128]
EPOCHS = 5
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])

for emb in embed_dims:
    print(f"\n=== Running embedding_dim={emb} ===")
    model = GRUClassifier(
        len(vocab), emb_dim=emb, hidden=64, num_labels=len(label_set)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # training
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
        val_loss, val_acc, val_swa, val_cwa, val_nrgs, _, _ = evaluate(
            model, dev_loader, criterion, train_signatures
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        run_record["losses"]["train"].append(train_loss)
        run_record["losses"]["val"].append(val_loss)
        run_record["metrics"]["train"].append({"epoch": epoch})
        run_record["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )
    # test evaluation
    test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues = evaluate(
        model, test_loader, criterion, train_signatures
    )
    print(
        f"Test: loss={test_loss:.4f} acc={test_acc:.3f} SWA={test_swa:.3f} CWA={test_cwa:.3f} NRGS={test_nrgs:.3f}"
    )
    run_record["losses"]["test"] = test_loss
    run_record["metrics"]["test"] = {
        "acc": test_acc,
        "swa": test_swa,
        "cwa": test_cwa,
        "nrgs": test_nrgs,
    }
    run_record["predictions"] = preds
    run_record["ground_truth"] = trues
    experiment_data["embedding_dim"]["SPR_BENCH"]["runs"][emb] = run_record
    # cleanup
    del model
    torch.cuda.empty_cache()

# ------------------------ save artefacts ------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))

# ------------------------ quick visualisation -------------------
dims = list(experiment_data["embedding_dim"]["SPR_BENCH"]["runs"].keys())
accs = [
    experiment_data["embedding_dim"]["SPR_BENCH"]["runs"][d]["metrics"]["test"]["acc"]
    for d in dims
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([str(d) for d in dims], accs, color="skyblue")
ax.set_ylim(0, 1)
ax.set_title("Test Accuracy vs Embedding Dim")
ax.set_xlabel("Embedding Dim")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plot_path = os.path.join(working_dir, "emb_dim_accuracy.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
