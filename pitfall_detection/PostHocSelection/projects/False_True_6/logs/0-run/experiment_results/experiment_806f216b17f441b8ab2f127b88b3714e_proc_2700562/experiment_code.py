import os, pathlib, math, time, json, random, itertools, numpy as np, torch, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------- reproducibility -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------- house-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- metrics ---------------------------------
def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def rule_signature(sequence):
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# --------------------------- data loading ----------------------------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# --------------------------- vocab & enc -----------------------------
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
def evaluate(model, loader, criterion):
    model.eval()
    total = correct = loss_sum = 0.0
    all_seq = []
    all_true = []
    all_pred = []
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
    novel_mask = [rule_signature(s) not in train_signatures for s in all_seq]
    novel_total = sum(novel_mask)
    novel_correct = sum(
        int(p == t) for p, t, m in zip(all_pred, all_true, novel_mask) if m
    )
    nrgs = novel_correct / novel_total if novel_total else 0.0
    return loss_sum / total, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# ----------------------- hyperparameter search -----------------------
weight_decays = [0.0, 1e-5, 5e-5, 1e-4, 5e-4]
EPOCHS = 5
experiment_data = {"weight_decay": {}}
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])
best_model_state = None
best_dev_acc = -1
best_tag = None

for wd in weight_decays:
    tag = f"wd_{wd}"
    print(f"\n==== Training with weight_decay={wd} ====")
    exp_entry = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
    model = GRUClassifier(len(vocab), emb=32, hidden=64, num_labels=len(label_set)).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader(128):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
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
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.3f}")
        exp_entry["SPR_BENCH"]["losses"]["train"].append(train_loss)
        exp_entry["SPR_BENCH"]["losses"]["val"].append(val_loss)
        exp_entry["SPR_BENCH"]["metrics"]["train"].append({"epoch": epoch})
        exp_entry["SPR_BENCH"]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )
        exp_entry["SPR_BENCH"]["timestamps"].append(time.time())
        # track best model across grid
        if val_acc > best_dev_acc:
            best_dev_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_tag = tag
    experiment_data["weight_decay"][tag] = exp_entry
    del model, optimizer
    torch.cuda.empty_cache()

print(f"\nBest setting: {best_tag} with dev_acc={best_dev_acc:.3f}")

# --------------------------- final test eval -------------------------
best_model = GRUClassifier(len(vocab), emb=32, hidden=64, num_labels=len(label_set)).to(
    device
)
best_model.load_state_dict(best_model_state)
criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues, seqs = evaluate(
    best_model, test_loader, criterion
)
print(
    f"\nTEST RESULTS ({best_tag}) loss={test_loss:.4f} acc={test_acc:.3f} "
    f"SWA={test_swa:.3f} CWA={test_cwa:.3f} NRGS={test_nrgs:.3f}"
)

# store test metrics under best tag
experiment_data["weight_decay"][best_tag]["SPR_BENCH"]["metrics"]["test"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
    "cwa": test_cwa,
    "nrgs": test_nrgs,
}
experiment_data["weight_decay"][best_tag]["SPR_BENCH"]["predictions"] = preds
experiment_data["weight_decay"][best_tag]["SPR_BENCH"]["ground_truth"] = trues

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# quick visualisation
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    ["Acc", "SWA", "CWA", "NRGS"],
    [test_acc, test_swa, test_cwa, test_nrgs],
    color="skyblue",
)
ax.set_ylim(0, 1)
ax.set_title(f"SPR_BENCH Test Metrics ({best_tag})")
plt.tight_layout()
plot_path = os.path.join(working_dir, "spr_metrics_bar.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
