import os, random, string, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Working directory and device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------
# Utility: metrics
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
# Data loading (with synthetic fallback)
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _ld("train.csv"), _ld("dev.csv"), _ld("test.csv")
    return d


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 10))
        )

    def rule(seq):
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

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
    print("SPR_BENCH not found, generating synthetic data …")
    make_synthetic_dataset(root)

spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Tokenisation / vocab
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
num_labels = len(set(spr["train"]["label"]))
max_len = 20


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


# -----------------------------------------------------------
# Model
class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(self.emb(x))
        return self.fc(out.mean(dim=1))


# -----------------------------------------------------------
# Experiment logger
experiment_data = {"batch_size": {}}


# -----------------------------------------------------------
# Training / evaluation helpers
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
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
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
# Hyperparameter tuning over batch sizes
batch_sizes = [32, 64, 128, 256]
epochs = 5
for bs in batch_sizes:
    print(f"\n==== Training with batch_size={bs} ====")
    # dataloaders
    train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=bs, shuffle=True)
    dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=bs)
    test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=bs)
    # model, criterion, optimizer
    model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # logger init
    key = f"SPR_BENCH_bs{bs}"
    experiment_data["batch_size"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        tr_loss, tr_swa, tr_cwa, tr_hwa, _, _ = run_epoch(
            model, train_dl, criterion, optim
        )
        val_loss, val_swa, val_cwa, val_hwa, _, _ = run_epoch(
            model, dev_dl, criterion, None
        )
        ed = experiment_data["batch_size"][key]
        ed["losses"]["train"].append((epoch, tr_loss))
        ed["losses"]["val"].append((epoch, val_loss))
        ed["metrics"]["train"].append((epoch, tr_hwa))
        ed["metrics"]["val"].append((epoch, val_hwa))
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f}, HWA={val_hwa:.4f} (SWA {val_swa:.4f}, CWA {val_cwa:.4f})"
        )
    # final test evaluation
    _, _, _, test_hwa, test_y, test_pred = run_epoch(model, test_dl, criterion, None)
    experiment_data["batch_size"][key]["predictions"] = test_pred
    experiment_data["batch_size"][key]["ground_truth"] = test_y
    print(f"Test HWA (bs={bs}) = {test_hwa:.4f}")

# -----------------------------------------------------------
# Save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
