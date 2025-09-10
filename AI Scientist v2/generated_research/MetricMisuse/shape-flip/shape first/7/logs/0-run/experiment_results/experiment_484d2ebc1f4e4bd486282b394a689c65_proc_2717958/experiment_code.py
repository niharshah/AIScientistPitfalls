import os, random, string, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Working directory / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------
# Metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / max(swa + cwa, 1e-6)


# -----------------------------------------------------------
# Dataset helpers (load real or synthetic)
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _ld("train.csv")
    d["dev"] = _ld("dev.csv")
    d["test"] = _ld("test.csv")
    return d


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # label rule
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, fname):
        with open(os.path.join(path, fname), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rand_seq()
                f.write(f"{i},{s},{rule(s)}\n")

    os.makedirs(path, exist_ok=True)
    mk(n_train, "train.csv")
    mk(n_dev, "dev.csv")
    mk(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found → generating synthetic data …")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Vocab / encoding
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len, num_labels = 20, len(set(spr["train"]["label"]))


def encode(seq, vocab=vocab, max_len=max_len):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------
# Torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.y = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
            "raw": self.seq[idx],
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# -----------------------------------------------------------
# Model def
class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


# -----------------------------------------------------------
# Experiment logger container
experiment_data = {"num_epochs": {"SPR_BENCH": {"runs": []}}}  # hyperparameter tuned


# -----------------------------------------------------------
# Train / evaluate helpers
def run_epoch(model, dl, criterion, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    tot_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(-1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa, cwa, hwa = (
        shape_weighted_accuracy(seqs, y_true, y_pred),
        color_weighted_accuracy(seqs, y_true, y_pred),
        harmonic_weighted_accuracy(seqs, y_true, y_pred),
    )
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# -----------------------------------------------------------
# Hyperparameter tuning over epoch counts
epoch_grid = [5, 10, 15, 20, 25, 30]
patience = 5

for max_epochs in epoch_grid:
    print(f"\n=== Training with max_epochs={max_epochs} ===")
    model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_state, stall = -1.0, None, 0
    losses_tr, losses_val, mets_tr, mets_val = [], [], [], []

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_swa, tr_cwa, tr_hwa, _, _ = run_epoch(
            model, train_dl, criterion, optim
        )
        val_loss, val_swa, val_cwa, val_hwa, _, _ = run_epoch(model, dev_dl, criterion)
        losses_tr.append((epoch, tr_loss))
        losses_val.append((epoch, val_loss))
        mets_tr.append((epoch, tr_hwa))
        mets_val.append((epoch, val_hwa))
        print(
            f"  Epoch {epoch}/{max_epochs} – val_loss {val_loss:.4f} HWA {val_hwa:.4f}"
        )

        if val_hwa > best_val + 1e-6:
            best_val, best_state, stall = (
                val_hwa,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            stall += 1
        if stall >= patience:
            print("  Early stopping.")
            break

    model.load_state_dict(best_state)
    _, _, _, test_hwa, test_true, test_pred = run_epoch(model, test_dl, criterion)
    print(f"→ Test HWA (best epoch) = {test_hwa:.4f}")

    # log run
    experiment_data["num_epochs"]["SPR_BENCH"]["runs"].append(
        {
            "epochs": max_epochs,
            "best_val_hwa": best_val,
            "metrics": {"train": mets_tr, "val": mets_val},
            "losses": {"train": losses_tr, "val": losses_val},
            "predictions": test_pred,
            "ground_truth": test_true,
        }
    )

# -----------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
