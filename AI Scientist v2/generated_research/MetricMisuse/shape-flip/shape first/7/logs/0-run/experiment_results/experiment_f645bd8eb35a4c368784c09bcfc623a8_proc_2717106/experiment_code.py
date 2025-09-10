import os, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------- Repro and device ----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------- Metrics -------------------------------------
def count_shape_variety(sequence):
    return len(set(t[0] for t in sequence.split()))


def count_color_variety(sequence):
    return len(set(t[1] for t in sequence.split()))


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


# ------------- Data (load or synth) ------------------------
def load_spr_bench(root):
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _ld(k) for k in ["train", "dev", "test"]})


def make_synth(path, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 10))
        )

    def rule(seq):
        return int(count_shape_variety(seq) > count_color_variety(seq))

    def mk(n, f):
        lines = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            lines.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, f"{f}.csv"), "w") as h:
            h.write("\n".join(lines))

    os.makedirs(path, exist_ok=True)
    mk(n_train, "train")
    mk(n_dev, "dev")
    mk(n_test, "test")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )
):
    print("Creating synthetic SPR_BENCH …")
    make_synth(root)

spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ------------- Vocab / encoding ----------------------------
def build_vocab(ds):
    v = {"<pad>": 0, "<unk>": 1}
    for seq in ds["sequence"]:
        for tok in seq.split():
            if tok not in v:
                v[tok] = len(v)
    return v


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(t, vocab["<unk>"]) for t in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# ------------- Torch Dataset -------------------------------
class SPRTorch(Dataset):
    def __init__(self, ds):
        self.seq, self.y = ds["sequence"], ds["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
            "raw": self.seq[idx],
        }


# ------------- Model ---------------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.gru = nn.GRU(embed, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.fc(h.mean(1))


# ------------- Experiment container ------------------------
experiment_data = {
    "batch_size": {
        "SPR_BENCH": {
            "metrics": {},
            "losses": {},
            "predictions": [],
            "ground_truth": [],
            "best_bs": None,
        }
    }
}
num_labels = len(set(spr["train"]["label"]))
batch_grid = [16, 32, 64, 128]
epochs = 5
best_hwa = -1
best_state = None
best_bs = None


# ------------- Helper: run epoch ---------------------------
def run_epoch(model, dl, criterion, optim=None):
    train_mode = optim is not None
    model.train() if train_mode else model.eval()
    tot_loss = 0
    y_t, y_p, seqs = [], [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["input"])
        loss = criterion(out, batch["label"])
        if train_mode:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = out.argmax(1).detach().cpu().tolist()
        y_p += preds
        y_t += batch["label"].cpu().tolist()
        seqs += batch["raw"]
    avg = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_t, y_p)
    cwa = color_weighted_accuracy(seqs, y_t, y_p)
    hwa = harmonic_weighted_accuracy(seqs, y_t, y_p)
    return avg, hwa, swa, cwa, y_t, y_p


# ------------- Grid search -------------------------------
for bs in batch_grid:
    print(f"\n=== Tuning batch_size={bs} ===")
    train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=bs, shuffle=True)
    dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=bs)
    model = GRUModel(len(vocab), num_classes=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    m_list, l_list = [], []
    for ep in range(1, epochs + 1):
        tr_loss, tr_hwa, _, _, _, _ = run_epoch(model, train_dl, criterion, optim)
        dv_loss, dv_hwa, _, _, _, _ = run_epoch(model, dev_dl, criterion)
        m_list.append((ep, tr_hwa, dv_hwa))
        l_list.append((ep, tr_loss, dv_loss))
        print(f"Ep{ep} | train_HWA:{tr_hwa:.4f} dev_HWA:{dv_hwa:.4f}")
    experiment_data["batch_size"]["SPR_BENCH"]["metrics"][bs] = m_list
    experiment_data["batch_size"]["SPR_BENCH"]["losses"][bs] = l_list
    if dv_hwa > best_hwa:
        best_hwa = dv_hwa
        best_state = model.state_dict()
        best_bs = bs

experiment_data["batch_size"]["SPR_BENCH"]["best_bs"] = best_bs
print(f"\nBest batch_size={best_bs} with dev_HWA={best_hwa:.4f}")

# ------------- Test evaluation with best ------------------
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=best_bs, shuffle=True)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=best_bs)
model = GRUModel(len(vocab), num_classes=num_labels).to(device)
model.load_state_dict(best_state)
criterion = nn.CrossEntropyLoss()
test_loss, test_hwa, _, _, y_true, y_pred = run_epoch(model, test_dl, criterion)
print(f"Test HWA={test_hwa:.4f}")

experiment_data["batch_size"]["SPR_BENCH"]["predictions"] = y_pred
experiment_data["batch_size"]["SPR_BENCH"]["ground_truth"] = y_true

# ------------- Save ---------------------------------------
work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)
np.save(
    os.path.join(work_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to", os.path.join(work_dir, "experiment_data.npy"))
