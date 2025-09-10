# ------------------ SET-UP & GPU ------------------
import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(13)
np.random.seed(13)
torch.manual_seed(13)

# ------------------ DATA UTILITIES ------------------
SHAPES_POOL = list("ABCDE")
COLORS_POOL = list("rgbym")


def _write_csv(path: pathlib.Path, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence", "label"])
        w.writerows(rows)


def _generate_synthetic_split(n_rows: int, start_id: int = 0):
    rows = []
    for idx in range(start_id, start_id + n_rows):
        seq_len = random.randint(4, 12)
        seq_tokens = []
        for _ in range(seq_len):
            s = random.choice(SHAPES_POOL)
            c = random.choice(COLORS_POOL)
            seq_tokens.append(s + c)
        sequence = " ".join(seq_tokens)
        lbl = "even" if (len(set(t[0] for t in seq_tokens)) % 2 == 0) else "odd"
        rows.append((idx, sequence, lbl))
    return rows


def _create_synthetic_bench(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "train.csv", _generate_synthetic_split(3000, 0))
    _write_csv(root / "dev.csv", _generate_synthetic_split(800, 4000))
    _write_csv(root / "test.csv", _generate_synthetic_split(1200, 5000))
    print(f"Synthetic SPR_BENCH generated at {root.resolve()}")


def find_spr_root() -> pathlib.Path:
    """Locate SPR_BENCH or create a synthetic one if missing."""
    # 1) explicit env-var
    env = os.getenv("SPR_DIR")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)

    # 2) walk up parent dirs
    cwd = pathlib.Path.cwd()
    for cand in [cwd / "SPR_BENCH", *cwd.resolve().parents]:
        if (cand / "train.csv").exists():
            return cand if cand.name == "SPR_BENCH" else cand / "SPR_BENCH"

    # 3) nothing found ‑> create synthetic
    synth_root = pathlib.Path(working_dir) / "SPR_BENCH"
    _create_synthetic_bench(synth_root)
    return synth_root


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=os.path.join(working_dir, ".cache_dsets"),
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------ DATASET WRAPPER ------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, shape_list, color_list, max_len=30):
        self.data = hf_split
        self.tok2id = tok2id
        self.lab2id = lab2id
        self.shape_list = shape_list
        self.color_list = color_list
        self.max_len = max_len

    def __len__(self):  # noqa
        return len(self.data)

    def _encode_tokens(self, seq):
        ids = [self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.split()]
        ids = ids[: self.max_len]
        pad = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * pad, len(ids)

    def _encode_symbolic(self, seq):
        shapes = [tok[0] for tok in seq.split() if tok]
        colors = [tok[1] for tok in seq.split() if len(tok) > 1]
        vec = np.zeros(
            len(self.shape_list) + len(self.color_list) + 2, dtype=np.float32
        )
        for s in shapes:
            if s in self.shape_list:
                vec[self.shape_list.index(s)] += 1
        for c in colors:
            if c in self.color_list:
                vec[len(self.shape_list) + self.color_list.index(c)] += 1
        vec[-2] = count_shape_variety(seq)
        vec[-1] = len(set(colors))
        return vec

    def __getitem__(self, idx):  # noqa
        row = self.data[idx]
        seq = row["sequence"]
        tok_ids, seq_len = self._encode_tokens(seq)
        sym = self._encode_symbolic(seq)
        return {
            "input_ids": torch.tensor(tok_ids, dtype=torch.long),
            "lengths": torch.tensor(seq_len, dtype=torch.long),
            "sym_feats": torch.tensor(sym, dtype=torch.float32),
            "label": torch.tensor(self.lab2id[row["label"]], dtype=torch.long),
            "raw_seq": seq,
        }


# ------------------ MODEL ------------------
class HybridClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx, sym_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.neural_head = nn.Linear(hid_dim * 2, n_cls)
        self.sym_head = nn.Sequential(
            nn.Linear(sym_dim, 64), nn.ReLU(), nn.Linear(64, n_cls)
        )
        self.mix_param = nn.Parameter(torch.tensor(0.0))

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


# ------------------ PREP DATA ------------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)

specials = ["<pad>", "<unk>"]
vocab = set()
shape_set, color_set = set(), set()
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        vocab.add(tok)
        shape_set.add(tok[0])
        if len(tok) > 1:
            color_set.add(tok[1])
tok2id = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
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

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# ------------------ TRAIN / EVAL HELPERS ------------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss, seen = 0.0, 0
    preds_all, labels_all, seqs_all = [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"], batch["sym_feats"])
            loss = criterion(logits, batch["label"])
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            seen += bs
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(batch["label"].cpu().numpy())
            seqs_all.extend(batch["raw_seq"])
    avg_loss = tot_loss / seen
    y_true = [id2lab[i] for i in labels_all]
    y_pred = [id2lab[i] for i in preds_all]
    swa = shape_weighted_accuracy(seqs_all, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# ------------------ EXPERIMENT TRACKER ------------------
experiment_data = {
    "hybrid": {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ------------------ TRAINING LOOP ------------------
model = HybridClassifier(
    len(tok2id), 32, 64, len(labels), pad_idx, len(shapes) + len(colors) + 2
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_swa, best_state, no_improve, patience = -1.0, None, 0, 4
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_swa, _, _ = run_epoch(model, dev_loader, criterion)
    experiment_data["hybrid"]["losses"]["train"].append(tr_loss)
    experiment_data["hybrid"]["losses"]["val"].append(val_loss)
    experiment_data["hybrid"]["metrics"]["train"].append(tr_swa)
    experiment_data["hybrid"]["metrics"]["val"].append(val_swa)
    experiment_data["hybrid"]["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA={val_swa:.4f}")
    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= patience:
        print("Early stopping triggered.")
        break

# ------------------ TEST EVAL ------------------
model.load_state_dict(best_state)
test_loss, test_swa, y_true, y_pred = run_epoch(model, test_loader, criterion)
experiment_data["hybrid"]["metrics"]["test"] = test_swa
experiment_data["hybrid"]["predictions"] = y_pred
experiment_data["hybrid"]["ground_truth"] = y_true
print(f"\nTest SWA = {test_swa:.4f}")

# ------------------ SAVE RESULTS ------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to {working_dir}/experiment_data.npy")

# quick val-loss plot
plt.figure()
plt.plot(experiment_data["hybrid"]["losses"]["val"])
plt.title("Validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(os.path.join(working_dir, "val_loss_curve.png"))
plt.close()
print(f"Plot saved to {working_dir}/val_loss_curve.png")
