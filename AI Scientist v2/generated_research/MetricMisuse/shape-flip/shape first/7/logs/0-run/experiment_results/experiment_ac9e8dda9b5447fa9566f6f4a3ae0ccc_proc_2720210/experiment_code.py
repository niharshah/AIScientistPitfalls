import os, random, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir & device handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# top-level bookkeeping dict
experiment_data = {
    "neural": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "symbolic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "hybrid": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# ------------------------------------------------------------------
# metric helpers (only SWA used officially)
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) else 1e-6)


# ------------------------------------------------------------------
# dataset loading (generate synthetic fallback)
def load_spr_bench(root: str) -> DatasetDict:
    def _ld(fname):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, fname),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def synth_spr(path, n_train=2000, n_dev=500, n_test=500):
    os.makedirs(path, exist_ok=True)
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 9))
        )

    def label(seq):
        return int(count_shape_variety(seq) % 2 == 0)

    for split, n in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{label(s)}")
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("\n".join(rows))


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("No SPR_BENCH found – creating synthetic data")
    synth_spr(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# vocab & tokenisation
vocab = {"<pad>": 0, "<unk>": 1}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id = vocab["<pad>"]
max_len = 20


def encode(seq):
    ids = [vocab.get(t, "<unk>") if t in vocab else 1 for t in seq.split()][:max_len]
    ids += [pad_id] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
# torch Dataset
class SPRTorch(Dataset):
    def __init__(self, hfds):
        self.seq = hfds["sequence"]
        self.y = hfds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        y = self.y[idx]
        feat = [
            count_shape_variety(seq),  # symbolic feature 1
            len(set(tok[1] for tok in seq.split() if len(tok) > 1)),  # colours
            len(seq.split()),
        ]  # length
        return {
            "ids": torch.tensor(encode(seq), dtype=torch.long),
            "sym": torch.tensor(feat, dtype=torch.float),
            "label": torch.tensor(y, dtype=torch.long),
            "raw": seq,
        }


batch_size = 64
dl_train = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dl_dev = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
dl_test = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# ------------------------------------------------------------------
# model definitions
class NeuralBranch(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.emb(x)
        o, _ = self.gru(emb)
        return o.mean(1)  # [B,2*hid]


class SymbolicBranch(nn.Module):
    def __init__(self, in_dim=3, h=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    def __init__(self, neural_dim, sym_dim, num_cls=2):
        super().__init__()
        self.fc = nn.Linear(neural_dim + sym_dim, num_cls)

    def forward(self, x):
        return self.fc(x)


class HybridModel(nn.Module):
    def __init__(self, vocab_size, sym_hidden=8):
        super().__init__()
        self.neural = NeuralBranch(vocab_size)
        self.sym = SymbolicBranch(h=sym_hidden)
        self.clf = Classifier(neural_dim=256, sym_dim=sym_hidden)

    def forward(self, ids, sym_feats):
        n = self.neural(ids)
        s = self.sym(sym_feats)
        out = torch.cat([n, s], dim=-1)
        return self.clf(out)


class NeuralOnly(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.neural = NeuralBranch(vocab_size)
        self.clf = nn.Linear(256, 2)

    def forward(self, ids, sym):
        return self.clf(self.neural(ids))


class SymbolicOnly(nn.Module):
    def __init__(self, sym_hidden=8):
        super().__init__()
        self.sym = SymbolicBranch(h=sym_hidden)
        self.clf = nn.Linear(sym_hidden, 2)

    def forward(self, ids, sym):
        return self.clf(self.sym(sym))


# ------------------------------------------------------------------
# training / evaluation helpers
def run_epoch(model, dl, criterion, opt=None):
    train = opt is not None
    (model.train() if train else model.eval())
    tot_loss = 0
    seqs = []
    yt = []
    yp = []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["ids"], batch["sym"])
        loss = criterion(logits, batch["label"])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(1).detach().cpu().tolist()
        yp.extend(preds)
        yt.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, yt, yp)
    return avg_loss, swa, yt, yp


# ------------------------------------------------------------------
# utility to train a model variant
def train_variant(name, model, mlp_h=8, epochs=4):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        tr_loss, tr_swa, _, _ = run_epoch(model, dl_train, crit, optim)
        dv_loss, dv_swa, _, _ = run_epoch(model, dl_dev, crit)
        experiment_data[name]["losses"]["train"].append((ep, tr_loss))
        experiment_data[name]["losses"]["val"].append((ep, dv_loss))
        experiment_data[name]["metrics"]["train"].append((ep, tr_swa))
        experiment_data[name]["metrics"]["val"].append((ep, dv_swa))
        print(
            f"[{name}] Epoch {ep}: validation_loss = {dv_loss:.4f}, SWA = {dv_swa:.4f}"
        )
    ts_loss, ts_swa, gt, pred = run_epoch(model, dl_test, crit)
    print(f"[{name}] Test SWA = {ts_swa:.4f}")
    experiment_data[name]["predictions"] = pred
    experiment_data[name]["ground_truth"] = gt


# ------------------------------------------------------------------
# run experiments
train_variant("neural", NeuralOnly(len(vocab)))
train_variant("symbolic", SymbolicOnly())
train_variant("hybrid", HybridModel(len(vocab)))

# ------------------------------------------------------------------
# save all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
