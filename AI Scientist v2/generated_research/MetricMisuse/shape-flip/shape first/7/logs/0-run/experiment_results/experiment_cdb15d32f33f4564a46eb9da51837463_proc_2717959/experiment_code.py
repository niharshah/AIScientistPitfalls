import os, random, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Experiment registry ----------------------------------------------------------
experiment_data = {
    "learning_rate": {  # <== hyper-parameter being tuned
        "SPR_BENCH": {  # dataset
            # each lr value will get its own sub-dict at run-time
        }
    }
}

# -----------------------------------------------------------
# Working directory / device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------
# Metric helpers ---------------------------------------------------------------
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
# Dataset (load real or synth) -----------------------------------------------
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _ld(f"{split}.csv")
    return d


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # label rule for synthetic data
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

    os.makedirs(path, exist_ok=True)
    mk(n_train, "train.csv")
    mk(n_dev, "dev.csv")
    mk(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found – generating synthetic dataset.")
    make_synthetic_dataset(root)

spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Vocab / tokeniser ------------------------------------------------------------
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


def encode(seq):  # returns list[int] length=max_len
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


# -----------------------------------------------------------
# Torch Dataset & DataLoaders --------------------------------------------------
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
# Model definition -------------------------------------------------------------
class BaselineGRU(nn.Module):
    def __init__(self, vocab_sz, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        return self.fc(out.mean(1))


# -----------------------------------------------------------
# Training / evaluation helpers -----------------------------------------------
def run_epoch(model, dataloader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, y_true, y_pred, seqs = 0.0, [], [], []
    for batch in dataloader:
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
    n = len(dataloader.dataset)
    swa, cwa, hwa = (
        shape_weighted_accuracy(seqs, y_true, y_pred),
        color_weighted_accuracy(seqs, y_true, y_pred),
        harmonic_weighted_accuracy(seqs, y_true, y_pred),
    )
    return tot_loss / n, swa, cwa, hwa, y_true, y_pred


# -----------------------------------------------------------
# Hyper-parameter sweep --------------------------------------------------------
learning_rates = [3e-4, 1e-3, 3e-3]
num_epochs = 5
best_hwa, best_model_state, best_lr = -1, None, None

for lr in learning_rates:
    print(f"\n=== Training with learning rate {lr} ===")
    model = BaselineGRU(len(vocab), num_classes=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # prepare logging slots
    lr_key = f"lr={lr}"
    experiment_data["learning_rate"]["SPR_BENCH"][lr_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_swa, tr_cwa, tr_hwa, *_ = run_epoch(
            model, train_dl, criterion, optimizer
        )
        val_loss, val_swa, val_cwa, val_hwa, *_ = run_epoch(model, dev_dl, criterion)

        ed = experiment_data["learning_rate"]["SPR_BENCH"][lr_key]
        ed["losses"]["train"].append((epoch, tr_loss))
        ed["losses"]["val"].append((epoch, val_loss))
        ed["metrics"]["train"].append((epoch, tr_hwa))
        ed["metrics"]["val"].append((epoch, val_hwa))

        print(f"epoch {epoch}: val_loss={val_loss:.4f} HWA={val_hwa:.4f}")

    # test set evaluation after final epoch
    _, _, _, test_hwa, y_true, y_pred = run_epoch(model, test_dl, criterion)
    ed["predictions"], ed["ground_truth"] = y_pred, y_true
    print(f"Test HWA @ lr={lr}: {test_hwa:.4f}")

    if val_hwa > best_hwa:
        best_hwa, best_model_state, best_lr = (
            val_hwa,
            {k: v.cpu() for k, v in model.state_dict().items()},
            lr,
        )

print(f"\nBest dev HWA={best_hwa:.4f} achieved at lr={best_lr}")

# -----------------------------------------------------------
# Save logs --------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
