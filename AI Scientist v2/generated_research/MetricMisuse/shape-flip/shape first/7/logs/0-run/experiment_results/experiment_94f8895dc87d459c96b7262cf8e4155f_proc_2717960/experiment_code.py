import os, random, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# Reproducibility & device
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# ------------------------------------------------------------------
# Metrics
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1e-6)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1e-6)


def harmonic_weighted_accuracy(seqs, y_t, y_p):
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    return 2 * swa * cwa / max(swa + cwa, 1e-6)


# ------------------------------------------------------------------
# Data (load bench or create synthetic fallback)
def load_spr_bench(root):
    def _ld(fname):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, fname),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def make_synthetic_dataset(path, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(s):
        return int(count_shape_variety(s) > count_color_variety(s))

    os.makedirs(path, exist_ok=True)
    for n, f in [(n_train, "train.csv"), (n_dev, "dev.csv"), (n_test, "test.csv")]:
        with open(os.path.join(path, f), "w") as out:
            out.write("id,sequence,label\n")
            for i in range(n):
                seq = rand_seq()
                out.write(f"{i},{seq},{rule(seq)}\n")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("Dataset not found – synthesising")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------------
# Vocab / encoding
def build_vocab(ds):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in ds["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
# Torch dataset
class SPRTorch(Dataset):
    def __init__(self, hfds):
        self.seq, self.y = hfds["sequence"], hfds["label"]

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


# ------------------------------------------------------------------
# GRU model with configurable pooling
class GRUPool(nn.Module):
    def __init__(self, vocab_sz, embed_dim=64, hid=64, num_classes=2, pooling="mean"):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.pooling = pooling
        if pooling == "attn":
            self.attn = nn.Linear(hid * 2, 1, bias=False)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(self.emb(x))  # [B,L,2H]
        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "max":
            pooled = out.max(dim=1).values
        elif self.pooling == "last":
            pooled = out[:, -1, :]
        elif self.pooling == "attn":
            alpha = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # [B,L]
            pooled = (out * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError
        return self.fc(pooled)


# ------------------------------------------------------------------
# Training / evaluation helpers
def run_epoch(model, dl, criterion, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    tot_loss = 0
    y_t = []
    y_p = []
    seqs = []
    for batch in dl:
        inp = batch["input"].to(device)
        labels = batch["label"].to(device)
        logits = model(inp)
        loss = criterion(logits, labels)
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * labels.size(0)
        preds = logits.argmax(-1).detach().cpu().tolist()
        y_p += preds
        y_t += labels.cpu().tolist()
        seqs += batch["raw"]
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    hwa = harmonic_weighted_accuracy(seqs, y_t, y_p)
    return avg_loss, swa, cwa, hwa, y_t, y_p


# ------------------------------------------------------------------
# Hyperparameter tuning over pooling methods
pooling_methods = ["mean", "max", "last", "attn"]
epochs = 4
experiment_data = {"pooling_method": {}}
best_dev_hwa = -1
best_pool = None
best_test_pred = None
best_test_true = None

for pm in pooling_methods:
    print(f"\n=== Training with pooling: {pm} ===")
    model = GRUPool(
        len(vocab), num_classes=len(set(spr["train"]["label"])), pooling=pm
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        tr_loss, tr_swa, tr_cwa, tr_hwa, _, _ = run_epoch(
            model, train_dl, criterion, optim
        )
        dv_loss, dv_swa, dv_cwa, dv_hwa, _, _ = run_epoch(model, dev_dl, criterion)
        exp["losses"]["train"].append((ep, tr_loss))
        exp["losses"]["val"].append((ep, dv_loss))
        exp["metrics"]["train"].append((ep, tr_hwa))
        exp["metrics"]["val"].append((ep, dv_hwa))
        print(f"Ep {ep}: dev HWA={dv_hwa:.4f} (SWA {dv_swa:.3f},CWA {dv_cwa:.3f})")
    # dev selection
    final_dev_hwa = exp["metrics"]["val"][-1][1]
    if final_dev_hwa > best_dev_hwa:
        best_dev_hwa = final_dev_hwa
        best_pool = pm
        _, _, _, test_hwa, test_y, test_pred = run_epoch(model, test_dl, criterion)
        best_test_pred, best_test_true = test_pred, test_y
        print(f"--> New best pooling ({pm}) with test HWA {test_hwa:.4f}")
    experiment_data["pooling_method"][pm] = exp

# ------------------------------------------------------------------
# Save best predictions
experiment_data["pooling_method"]["best_pooling"] = best_pool
experiment_data["pooling_method"]["predictions"] = best_test_pred
experiment_data["pooling_method"]["ground_truth"] = best_test_true
print(f"\nBest pooling: {best_pool} | Dev HWA {best_dev_hwa:.4f}")

# ------------------------------------------------------------------
# persist
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data ->", os.path.join(working_dir, "experiment_data.npy"))
