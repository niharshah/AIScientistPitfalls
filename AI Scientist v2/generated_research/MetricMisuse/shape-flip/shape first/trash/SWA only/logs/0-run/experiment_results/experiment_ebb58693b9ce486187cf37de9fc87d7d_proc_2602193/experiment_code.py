# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, string, datetime, json, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# House-keeping & working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "dev": [], "test": [], "NRGS": []},
        "losses": {"train": [], "dev": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------------------------------------------------------------
# Device management (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# Attempt to load real SPR_BENCH or fall back to synthetic
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{split}.csv"))
        for split in ["train", "dev", "test"]
    )


use_synthetic = not spr_files_exist(SPR_PATH)

if use_synthetic:
    print("Real SPR_BENCH not found – generating synthetic data.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = [str(i) for i in range(4)]  # 0-3

    def random_seq():
        length = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )

    def rule_label(seq):
        # simple synthetic rule: 1 if #unique shapes == #unique colors else 0
        us = len(set(tok[0] for tok in seq.split()))
        uc = len(set(tok[1] for tok in seq.split()))
        return int(us == uc)

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        labels = [rule_label(s) for s in seqs]
        return {"sequence": seqs, "label": labels}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    print("Loading real SPR_BENCH")
    import pathlib
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root: str):
        def _load(split_csv):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, split_csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict()
        d["train"] = _load("train.csv")
        d["dev"] = _load("dev.csv")
        d["test"] = _load("test.csv")
        return d

    ds = load_spr_bench(SPR_PATH)
    raw_data = {
        split: {"sequence": ds[split]["sequence"], "label": ds[split]["label"]}
        for split in ["train", "dev", "test"]
    }


# ---------------------------------------------------------------------
# Helper metrics
def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) or 1)


# NRGS calculation
def compute_signatures(seqs):
    sigs = []
    for s in seqs:
        shapes = tuple(sorted(set(tok[0] for tok in s.split())))
        colors = tuple(sorted(set(tok[1] for tok in s.split())))
        sigs.append((shapes, colors))
    return sigs


# ---------------------------------------------------------------------
# Tokenizer / vocab
PAD = "<PAD>"
UNK = "<UNK>"


def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------------------------------------------------------------------
# PyTorch dataset
class SPRTorchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = [torch.tensor(encode_sequence(s), dtype=torch.long) for s in sequences]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw_seq = sequences

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "label": self.y[idx]}


def collate(batch):
    lengths = [len(item["input_ids"]) for item in batch]
    maxlen = max(lengths)
    input_ids = torch.full(
        (len(batch), maxlen), fill_value=vocab[PAD], dtype=torch.long
    )
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
        labels[i] = item["label"]
    return {"input_ids": input_ids, "labels": labels, "lengths": torch.tensor(lengths)}


datasets = {
    split: SPRTorchDataset(raw_data[split]["sequence"], raw_data[split]["label"])
    for split in ["train", "dev", "test"]
}


# ---------------------------------------------------------------------
# Model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        logits = self.out(h.squeeze(0))
        return logits


num_classes = len(set(raw_data["train"]["label"]))
model = GRUClassifier(
    vocab_size, embed_dim=64, hidden_dim=128, num_classes=num_classes
).to(device)

# Optimizer & loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# DataLoaders
batch_size = 64
loaders = {
    split: DataLoader(
        datasets[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
    )
    for split in ["train", "dev", "test"]
}

# ---------------------------------------------------------------------
# Training loop
epochs = 6
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in loaders["train"]:
        # move tensors
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    avg_train_loss = running_loss / len(datasets["train"])
    experiment_data["SPR_BENCH"]["losses"]["train"].append(avg_train_loss)

    # ------------------ validation
    model.eval()

    def evaluate(split):
        correct, total, loss_sum = 0, 0, 0
        all_seq, y_true, y_pred = [], [], []
        with torch.no_grad():
            for batch in loaders[split]:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"], batch["lengths"])
                loss = criterion(logits, batch["labels"])
                preds = logits.argmax(-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                loss_sum += loss.item() * batch["labels"].size(0)
                seq_idx = loaders[split].dataset.raw_seq
            # We need raw sequences aligned with predictions; easier by iterating again:
        all_seq = loaders[split].dataset.raw_seq
        y_true = loaders[split].dataset.y.tolist()
        # recompute preds for all_seq (small cost)
        pred_list = []
        with torch.no_grad():
            for i in range(0, len(all_seq), batch_size):
                batch_seqs = all_seq[i : i + batch_size]
                enc = [encode_sequence(s) for s in batch_seqs]
                lengths = torch.tensor([len(x) for x in enc])
                maxlen = lengths.max()
                inp = torch.full((len(enc), maxlen), vocab[PAD], dtype=torch.long)
                for j, row in enumerate(enc):
                    inp[j, : len(row)] = torch.tensor(row)
                logits = model(inp.to(device), lengths.to(device))
                pred_list.extend(logits.argmax(-1).cpu().tolist())
        acc = correct / total
        swa = shape_weighted_accuracy(all_seq, y_true, pred_list)
        cwa = color_weighted_accuracy(all_seq, y_true, pred_list)
        return acc, swa, cwa, loss_sum / total, pred_list, y_true, all_seq

    dev_acc, dev_swa, dev_cwa, dev_loss, _, _, _ = evaluate("dev")
    experiment_data["SPR_BENCH"]["losses"]["dev"].append(dev_loss)
    experiment_data["SPR_BENCH"]["metrics"]["dev"].append(
        {"acc": dev_acc, "swa": dev_swa, "cwa": dev_cwa}
    )
    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={dev_loss:.4f}  val_acc={dev_acc:.3f}"
    )

    experiment_data["SPR_BENCH"]["timestamps"].append(str(datetime.datetime.now()))

# ---------------------------------------------------------------------
# Final test evaluation & NRGS
test_acc, test_swa, test_cwa, _, preds, gts, seqs = evaluate("test")
print(f"TEST  acc={test_acc:.3f}  SWA={test_swa:.3f}  CWA={test_cwa:.3f}")

# NRGS
train_sigs = set(compute_signatures(raw_data["train"]["sequence"]))
test_sigs = compute_signatures(seqs)
novel_idx = [i for i, sg in enumerate(test_sigs) if sg not in train_sigs]
if novel_idx:
    novel_correct = sum(1 for i in novel_idx if preds[i] == gts[i])
    NRGS = novel_correct / len(novel_idx)
else:
    NRGS = 0.0
print(f"Novel Rule Generalization Score (NRGS): {NRGS:.3f}")

# Populate experiment_data
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "acc": test_acc,
    "swa": test_swa,
    "cwa": test_cwa,
}
experiment_data["SPR_BENCH"]["metrics"]["NRGS"] = NRGS
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# ---------------------------------------------------------------------
# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

# ---------------------------------------------------------------------
# Visualization – loss curves
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["dev"], label="dev")
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_SPR.png"))
plt.close()
