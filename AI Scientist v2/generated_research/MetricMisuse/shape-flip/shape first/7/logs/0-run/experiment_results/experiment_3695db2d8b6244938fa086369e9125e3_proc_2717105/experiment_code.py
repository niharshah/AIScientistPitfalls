import os, random, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------- experiment data container ------------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "all_runs": [],  # store every lr sweep
        }
    }
}

# ------------- device & reproducibility -------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------- metric helpers -----------------------------
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


# ------------- data loading -------------------------------
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _ld("train.csv"),
            "dev": _ld("dev.csv"),
            "test": _ld("test.csv"),
        }
    )


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 10))
        )

    def rule(seq):  # label rule
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def write(n, fname):
        with open(os.path.join(path, fname), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rand_seq()
                f.write(f"{i},{s},{rule(s)}\n")

    os.makedirs(path, exist_ok=True)
    write(n_train, "train.csv")
    write(n_dev, "dev.csv")
    write(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )
):
    print("Real SPR_BENCH not found – generating synthetic data.")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ------------- vocab / encoding ---------------------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len, num_labels = 20, len(set(spr["train"]["label"]))


def encode(seq):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


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


# ------------- model definition ---------------------------
class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        pooled = out.mean(1)
        return self.fc(pooled)


# ------------- training helpers ---------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dl, optimizer=None):
    train_mode = optimizer is not None
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(-1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    L = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return L, swa, cwa, hwa, y_true, y_pred


# ------------- hyper-parameter sweep ----------------------
lrs = [2e-3, 1e-3, 5e-4, 3e-4]
epochs = 5
best_val, best_artifact = -1, None

for lr in lrs:
    model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    run_record = {
        "lr": lr,
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }
    print(f"\n=== Training with learning rate {lr:.0e} ===")
    for ep in range(1, epochs + 1):
        tr_loss, *_, tr_hwa, _, _ = run_epoch(model, train_dl, optim)
        val_loss, *_, val_hwa, _, _ = run_epoch(model, dev_dl)
        run_record["losses"]["train"].append((ep, tr_loss))
        run_record["losses"]["val"].append((ep, val_loss))
        run_record["metrics"]["train"].append((ep, tr_hwa))
        run_record["metrics"]["val"].append((ep, val_hwa))
        print(f"  Epoch {ep}: val_loss={val_loss:.4f}  val_HWA={val_hwa:.4f}")
        if val_hwa > best_val:
            best_val = val_hwa
            best_artifact = {"lr": lr, "epoch": ep, "model_state": model.state_dict()}
    experiment_data["learning_rate"]["SPR_BENCH"]["all_runs"].append(run_record)

# ------------- restore best model & final test ------------
print(
    f"\nBest validation HWA {best_val:.4f} achieved with lr={best_artifact['lr']:.0e} at epoch {best_artifact['epoch']}"
)
best_model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
best_model.load_state_dict(best_artifact["model_state"])
test_loss, _, _, test_hwa, test_y, test_pred = run_epoch(best_model, test_dl)
print(f"Test HWA = {test_hwa:.4f}")

# store aggregated (best run) info
best_run_idx = next(
    i
    for i, r in enumerate(experiment_data["learning_rate"]["SPR_BENCH"]["all_runs"])
    if abs(r["lr"] - best_artifact["lr"]) < 1e-12
)
experiment_data["learning_rate"]["SPR_BENCH"]["metrics"] = experiment_data[
    "learning_rate"
]["SPR_BENCH"]["all_runs"][best_run_idx]["metrics"]
experiment_data["learning_rate"]["SPR_BENCH"]["losses"] = experiment_data[
    "learning_rate"
]["SPR_BENCH"]["all_runs"][best_run_idx]["losses"]
experiment_data["learning_rate"]["SPR_BENCH"]["predictions"] = test_pred
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = test_y

# ------------- save outputs -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
