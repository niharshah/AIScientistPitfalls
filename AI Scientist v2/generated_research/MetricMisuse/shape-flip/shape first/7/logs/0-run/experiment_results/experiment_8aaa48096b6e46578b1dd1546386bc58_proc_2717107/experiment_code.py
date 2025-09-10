import os, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------
# Working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------
# Metrics
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1e-6)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / max(sum(w), 1e-6)


def harmonic_weighted_accuracy(seqs, y_t, y_p):
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    return 2 * swa * cwa / max(swa + cwa, 1e-6)


# -----------------------------------------------------------
# Data loading / synthetic fallback
def load_spr_bench(root):
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))


def make_synthetic_dataset(path, n_train=2000, n_dev=500, n_test=500):
    shapes = list("STCH")
    colors = list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, f):
        with open(os.path.join(path, f), "w") as fp:
            fp.write("id,sequence,label\n")
            for i in range(n):
                s = rand_seq()
                fp.write(f"{i},{s},{rule(s)}\n")

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
    print("Generating synthetic SPR_BENCH …")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Vocabulary & encoding
def build_vocab(ds):
    v = {"<pad>": 0, "<unk>": 1}
    for seq in ds["sequence"]:
        for tok in seq.split():
            if tok not in v:
                v[tok] = len(v)
    return v


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq, vocab=vocab, max_len=max_len):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------
# Torch dataset
class SPRTorch(Dataset):
    def __init__(self, hfds):
        self.seq = hfds["sequence"]
        self.y = hfds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
            "raw": self.seq[idx],
        }


batch_size = 64
train_dl_all = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dev_dl_all = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl_all = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# -----------------------------------------------------------
# Model definition
class GRUClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.gru(x)
        pooled = out.mean(1)
        return self.fc(pooled)


# -----------------------------------------------------------
# Training / evaluation helpers
def run_epoch(model, dl, criterion, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    tot_loss = 0
    y_t = []
    y_p = []
    seqs = []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(-1).detach().cpu().tolist()
        y_p.extend(preds)
        y_t.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    hwa = harmonic_weighted_accuracy(seqs, y_t, y_p)
    return avg_loss, swa, cwa, hwa, y_t, y_p


# -----------------------------------------------------------
# Hyperparameter sweep
hidden_sizes = [32, 64, 128, 256]
epochs = 5
num_labels = len(set(spr["train"]["label"]))

experiment_data = {"hidden_size": {}}

for hid in hidden_sizes:
    print(f"\n--- Training with hidden_size={hid} ---")
    model = GRUClassifier(len(vocab), hid=hid, num_classes=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist_loss_tr, hist_loss_val, hist_hwa_tr, hist_hwa_val = [], [], [], []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_swa, tr_cwa, tr_hwa, _, _ = run_epoch(
            model, train_dl_all, criterion, optimizer
        )
        val_loss, val_swa, val_cwa, val_hwa, _, _ = run_epoch(
            model, dev_dl_all, criterion
        )
        hist_loss_tr.append((epoch, tr_loss))
        hist_loss_val.append((epoch, val_loss))
        hist_hwa_tr.append((epoch, tr_hwa))
        hist_hwa_val.append((epoch, val_hwa))
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, HWA={val_hwa:.4f}")

    # final test evaluation
    _, _, _, test_hwa, test_y, test_pred = run_epoch(model, test_dl_all, criterion)
    print(f"hidden_size={hid} Test HWA: {test_hwa:.4f}")

    experiment_data["hidden_size"][str(hid)] = {
        "metrics": {"train": hist_hwa_tr, "val": hist_hwa_val},
        "losses": {"train": hist_loss_tr, "val": hist_loss_val},
        "predictions": test_pred,
        "ground_truth": test_y,
        "test_hwa": test_hwa,
    }
    torch.cuda.empty_cache()

# -----------------------------------------------------------
# Save
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All experiments saved to", os.path.join(working_dir, "experiment_data.npy"))
