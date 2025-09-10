# ==============================  hyperparameter_tuning_hidden_dim.py  ==============================
import os, random, string, datetime, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. House-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "hidden_dim_tuning": {
        "SPR_BENCH": {}  # we add one sub-dict per hidden_dim value below
    }
}

# ---------------------------------------------------------------------
# 2. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# 3. Data  (load real SPR_BENCH if available, otherwise generate synthetic)
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
        # simple rule: 1 iff #unique shapes == #unique colors
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
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root: str):
        def _load(fname):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, fname),
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
# 4. Metrics helpers
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


def compute_signatures(seqs):
    sigs = []
    for s in seqs:
        shapes = tuple(sorted(set(tok[0] for tok in s.split())))
        colors = tuple(sorted(set(tok[1] for tok in s.split())))
        sigs.append((shapes, colors))
    return sigs


# ---------------------------------------------------------------------
# 5. Tokenizer / vocab
PAD, UNK = "<PAD>", "<UNK>"


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
# 6. DataSet & DataLoader
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


base_datasets = {
    split: SPRTorchDataset(raw_data[split]["sequence"], raw_data[split]["label"])
    for split in ["train", "dev", "test"]
}


# ---------------------------------------------------------------------
# 7. Model definition
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


# ---------------------------------------------------------------------
# 8. Training & evaluation helpers
def evaluate(model, loaders, split, criterion, batch_size):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
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

    # full prediction list needed for weighted metrics & NRGS
    seqs = loaders[split].dataset.raw_seq
    gts = loaders[split].dataset.y.tolist()
    preds_all = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            enc = [encode_sequence(s) for s in chunk]
            lengths = torch.tensor([len(x) for x in enc])
            maxlen = lengths.max()
            inp = torch.full((len(enc), maxlen), vocab[PAD], dtype=torch.long)
            for j, row in enumerate(enc):
                inp[j, : len(row)] = torch.tensor(row)
            logits = model(inp.to(device), lengths.to(device))
            preds_all.extend(logits.argmax(-1).cpu().tolist())

    acc = correct / total
    swa = shape_weighted_accuracy(seqs, gts, preds_all)
    cwa = color_weighted_accuracy(seqs, gts, preds_all)
    return acc, swa, cwa, loss_sum / total, preds_all, gts, seqs


def train_single_setting(hidden_dim, epochs=6, batch_size=64, lr=1e-3):
    print(f"\n=======  Training with hidden_dim={hidden_dim}  =======")
    data_loaders = {
        split: DataLoader(
            base_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "dev", "test"]
    }
    model = GRUClassifier(
        vocab_size=vocab_size, embed_dim=64, hidden_dim=hidden_dim, num_classes=2
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses_train, losses_dev = [], []
    metrics_dev = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in data_loaders["train"]:
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
        avg_train_loss = running_loss / len(base_datasets["train"])
        losses_train.append(avg_train_loss)

        # dev evaluation
        dev_acc, dev_swa, dev_cwa, dev_loss, *_ = evaluate(
            model, data_loaders, "dev", criterion, batch_size
        )
        losses_dev.append(dev_loss)
        metrics_dev.append({"acc": dev_acc, "swa": dev_swa, "cwa": dev_cwa})
        print(
            f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}  "
            f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.3f}"
        )

    # final test evaluation & NRGS
    test_acc, test_swa, test_cwa, _, preds, gts, seqs = evaluate(
        model, data_loaders, "test", criterion, batch_size
    )
    train_sigs = set(compute_signatures(raw_data["train"]["sequence"]))
    test_sigs = compute_signatures(seqs)
    novel_idx = [i for i, sg in enumerate(test_sigs) if sg not in train_sigs]
    NRGS = (
        (sum(1 for i in novel_idx if preds[i] == gts[i]) / len(novel_idx))
        if novel_idx
        else 0.0
    )
    print(
        f"  TEST  acc={test_acc:.3f}  SWA={test_swa:.3f}  CWA={test_cwa:.3f}  NRGS={NRGS:.3f}"
    )

    # collect everything
    setting_data = {
        "metrics": {
            "train": [],  # not used, kept for compatibility
            "dev": metrics_dev,
            "test": {"acc": test_acc, "swa": test_swa, "cwa": test_cwa},
            "NRGS": NRGS,
        },
        "losses": {"train": losses_train, "dev": losses_dev},
        "predictions": preds,
        "ground_truth": gts,
        "timestamps": [str(datetime.datetime.now())],
    }
    return setting_data, model


# ---------------------------------------------------------------------
# 9. Hyper-parameter tuning loop
hidden_dim_values = [64, 128, 256]
best_dev_acc, best_setting = -1.0, None

for hd in hidden_dim_values:
    data_dict, _ = train_single_setting(hd)
    experiment_data["hidden_dim_tuning"]["SPR_BENCH"][f"hidden_dim={hd}"] = data_dict
    last_dev_acc = data_dict["metrics"]["dev"][-1]["acc"]
    if last_dev_acc > best_dev_acc:
        best_dev_acc, best_setting = last_dev_acc, hd

print(f"\nBest hidden_dim on dev set: {best_setting}  (acc={best_dev_acc:.3f})")

# ---------------------------------------------------------------------
# 10. Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

# 11. Plot loss curves for each hidden_dim
for hd in hidden_dim_values:
    losses_t = experiment_data["hidden_dim_tuning"]["SPR_BENCH"][f"hidden_dim={hd}"][
        "losses"
    ]["train"]
    losses_d = experiment_data["hidden_dim_tuning"]["SPR_BENCH"][f"hidden_dim={hd}"][
        "losses"
    ]["dev"]
    plt.figure()
    plt.plot(losses_t, label="train")
    plt.plot(losses_d, label="dev")
    plt.title(f"Loss (hidden_dim={hd})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_hd_{hd}.png"))
    plt.close()

print("\nAll results saved to 'working/experiment_data.npy' and JSON/PNG companions.")
