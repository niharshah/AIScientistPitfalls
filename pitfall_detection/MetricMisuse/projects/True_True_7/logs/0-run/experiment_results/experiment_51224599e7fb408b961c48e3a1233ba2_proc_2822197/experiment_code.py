import os, pathlib, time, random, math, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------- experiment data container -------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------- synthetic data helper -----------------
SHAPES = ["A", "B", "C", "D"]
COLORS = ["0", "1", "2"]
LABELS = ["yes", "no"]


def _rand_token():
    s = random.choice(SHAPES)
    if random.random() < 0.7:  # 70% tokens have color
        s += random.choice(COLORS)
    return s


def _rand_sequence(min_len=3, max_len=10):
    return " ".join(_rand_token() for _ in range(random.randint(min_len, max_len)))


def _gen_split(n_rows):
    rows = []
    for i in range(n_rows):
        seq = _rand_sequence()
        lbl = random.choice(LABELS)
        rows.append(f"{i},{seq},{lbl}\n")
    return ["id,sequence,label\n"] + rows


def _create_synthetic_bench(root: pathlib.Path):
    print("Creating synthetic SPR_BENCH in", root)
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train.csv": 300, "dev.csv": 60, "test.csv": 100}
    for fname, n in splits.items():
        with open(root / fname, "w") as f:
            f.writelines(_gen_split(n))


# ------------- locate dataset -----------------
def find_spr_root() -> pathlib.Path:
    """Return a folder containing train/dev/test csv; create synthetic if needed."""
    env = os.getenv("SPR_DIR")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    # search upward
    here = pathlib.Path.cwd()
    for cand in [here, *(here.parents)]:
        if (cand / "SPR_BENCH" / "train.csv").exists():
            return cand / "SPR_BENCH"
    # fallback: create synthetic data inside working dir
    synth_root = pathlib.Path(working_dir) / "SPR_BENCH"
    _create_synthetic_bench(synth_root)
    return synth_root


# ------------- load dataset -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(root / name),
            split="train",
            cache_dir=os.path.join(working_dir, ".cache_dsets"),
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------------- dataset class ----------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, shapes, colors, max_len=30):
        self.data = hf_split
        self.tok2id = tok2id
        self.lab2id = lab2id
        self.shapes = shapes
        self.colors = colors
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode_tokens(self, seq):
        ids = [self.tok2id.get(t, self.tok2id["<unk>"]) for t in seq.split()]
        ids = ids[: self.max_len]
        pad = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * pad, len(ids)

    def _encode_symbolic(self, seq):
        shapes = [tok[0] for tok in seq.split() if tok]
        colors = [tok[1] for tok in seq.split() if len(tok) > 1]
        vec = np.zeros(len(self.shapes) + len(self.colors) + 2, dtype=np.float32)
        for s in shapes:
            if s in self.shapes:
                vec[self.shapes.index(s)] += 1
        for c in colors:
            if c in self.colors:
                vec[len(self.shapes) + self.colors.index(c)] += 1
        vec[-2] = count_shape_variety(seq)
        vec[-1] = count_color_variety(seq)
        return vec

    def __getitem__(self, idx):
        row = self.data[idx]
        seq = row["sequence"]
        ids, length = self._encode_tokens(seq)
        sym = self._encode_symbolic(seq)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "sym_feats": torch.tensor(sym, dtype=torch.float32),
            "label": torch.tensor(self.lab2id[row["label"]], dtype=torch.long),
            "raw_seq": seq,
        }


# ---------------- model ----------------
class HybridClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx, sym_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.neural_head = nn.Linear(hid_dim * 2, n_cls)
        self.sym_head = nn.Sequential(
            nn.Linear(sym_dim, 64), nn.ReLU(), nn.Linear(64, n_cls)
        )
        self.mix_param = nn.Parameter(torch.tensor(0.0))  # learned α (sigmoid)

    def forward(self, tok_ids, lengths, sym_feats):
        emb = self.emb(tok_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        h = out.gather(1, idx).squeeze(1)
        neural_logits = self.neural_head(h)
        sym_logits = self.sym_head(sym_feats)
        alpha = torch.sigmoid(self.mix_param)
        return (1 - alpha) * neural_logits + alpha * sym_logits


# ---------------- prepare data ----------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

specials = ["<pad>", "<unk>"]
vocab_set, shape_set, color_set = set(), set(), set()
for s in spr["train"]["sequence"]:
    for tok in s.split():
        vocab_set.add(tok)
        shape_set.add(tok[0])
        if len(tok) > 1:
            color_set.add(tok[1])

tok2id = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab_set))}
for i, sp in enumerate(specials):
    tok2id[sp] = i
pad_idx = tok2id["<pad>"]
shapes, colors = sorted(shape_set), sorted(color_set)

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}

train_ds = SPRDataset(spr["train"], tok2id, lab2id, shapes, colors)
dev_ds = SPRDataset(spr["dev"], tok2id, lab2id, shapes, colors)
test_ds = SPRDataset(spr["test"], tok2id, lab2id, shapes, colors)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)


# ---------------- train / eval helpers ----------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, tot = 0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train_flag):
        for batch in loader:
            # move tensors to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"], batch["sym_feats"])
            loss = criterion(logits, batch["label"])
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            tot += bs
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = tot_loss / tot
    y_true = [id2lab[i] for i in all_labels]
    y_pred = [id2lab[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# ---------------- training loop ----------------
model = HybridClassifier(
    len(tok2id), 32, 64, len(labels), pad_idx, len(shapes) + len(colors) + 2
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs, patience = 8, 3
best_val_swa, best_state, no_imp = -1.0, None, 0
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_swa, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_swa, _, _ = run_epoch(model, dev_loader, criterion)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA={val_swa:.4f}")
    if val_swa > best_val_swa:
        best_val_swa, best_state, no_imp = (
            val_swa,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        no_imp += 1
    if no_imp >= patience:
        print("Early stopping.")
        break

# ---------------- test evaluation ----------------
model.load_state_dict(best_state)
test_loss, test_swa, y_true, y_pred = run_epoch(model, test_loader, criterion)
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa
experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
print(f"\nTest SWA = {test_swa:.4f}")

# ---------------- save artefacts -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}/experiment_data.npy")

plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"])
plt.title("Val loss (SPR_BENCH)")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(working_dir, "SPR_val_loss_curve.png"))
plt.close()
