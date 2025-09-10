import os, random, torch, numpy as np, pathlib, time
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------- #
# 0.  House-keeping & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
disable_caching()  # keep HF cache small
torch.backends.cudnn.benchmark = True


# -------------------------------------------------- #
# 1.  Helper metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-8)


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-8)


# -------------------------------------------------- #
# 2.  Synthetic data generation
shapes, colors = list("STCH"), list("RGBY")


def rnd_seq():
    L = random.randint(3, 9)
    return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))


def make_dataset(path: str, rule_fn, seed: int, n_tr=6000, n_dev=1500, n_te=1500):
    splits = [("train", n_tr), ("dev", n_dev), ("test", n_te)]
    if os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, f"{s}.csv")) or n == 0 for s, n in splits
    ):
        return  # already present
    print("Creating synthetic data at", path)
    os.makedirs(path, exist_ok=True)
    random.seed(seed)
    for split, n in splits:
        if n == 0:  # skip empty splits (avoid buggy 0-row files)
            # ensure no stale file exists
            fpath = os.path.join(path, f"{split}.csv")
            if os.path.isfile(fpath):
                os.remove(fpath)
            continue
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = rnd_seq()
                sh = len(set(t[0] for t in seq.split()))
                co = len(set(t[1] for t in seq.split()))
                f.write(f"{i},{seq},{rule_fn(sh, co)}\n")


def load_spr(root: str, splits=("train", "dev", "test")) -> DatasetDict:
    d = DatasetDict()
    for split in splits:
        fpath = os.path.join(root, f"{split}.csv")
        if not os.path.isfile(fpath):
            continue  # split missing (e.g., hold-out has only test)
        d[split] = load_dataset(
            "csv",
            data_files=fpath,
            split="train",
            cache_dir=".cache_dsets",
        )
    return d


# -------------------------------------------------- #
# 3.  Dataset variants + hold-out
variants = [
    ("delta0_gt", lambda sh, co: int(sh > co), 11),  # shapes >  colours
    ("delta0_ge", lambda sh, co: int(sh >= co), 22),  # shapes ≥ colours
    (
        "delta1_gt1",
        lambda sh, co: int((sh - co) > 1),
        33,
    ),  # shapes exceed colours by >1
]
holdout_name, holdout_rule, holdout_seed = "holdout", lambda sh, co: int(sh > co), 77

for name, rule, seed in variants:
    make_dataset(name, rule, seed)
make_dataset(holdout_name, holdout_rule, holdout_seed, n_tr=0, n_dev=0, n_te=2000)

# -------------------------------------------------- #
# 4.  Vocab & encoding
shape_vocab = {s: i + 1 for i, s in enumerate(sorted(shapes))}
color_vocab = {c: i + 1 for i, c in enumerate(sorted(colors))}
pad_idx, max_len = 0, 20


def encode(seq: str):
    s_ids, c_ids = [], []
    for tok in seq.split()[:max_len]:
        s_ids.append(shape_vocab.get(tok[0], 0))
        c_ids.append(color_vocab.get(tok[1], 0))
    while len(s_ids) < max_len:
        s_ids.append(pad_idx)
        c_ids.append(pad_idx)
    return s_ids, c_ids


# -------------------------------------------------- #
# 5.  PyTorch Dataset
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [int(x) for x in hf_split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        s_ids, c_ids = encode(seq)
        sym_feats = torch.tensor(
            [count_shape_variety(seq), count_color_variety(seq)], dtype=torch.float
        )
        return {
            "shape": torch.tensor(s_ids, dtype=torch.long),
            "color": torch.tensor(c_ids, dtype=torch.long),
            "sym": sym_feats,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


# -------------------------------------------------- #
# 6.  Model
class DisentangledTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=2, sym_dim=2, n_cls=2, dropout=0.1):
        super().__init__()
        self.shape_emb = nn.Embedding(
            len(shape_vocab) + 1, d_model, padding_idx=pad_idx
        )
        self.color_emb = nn.Embedding(
            len(color_vocab) + 1, d_model, padding_idx=pad_idx
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_cls)

    def forward(self, sh_ids, col_ids, sym_feats):
        tok_emb = self.shape_emb(sh_ids) + self.color_emb(col_ids)
        enc = self.encoder(tok_emb)
        pooled = enc.mean(1)
        sym_emb = torch.relu(self.sym_fc(sym_feats))
        logits = self.out(torch.cat([pooled, sym_emb], dim=-1))
        return logits


# -------------------------------------------------- #
# 7.  Train / evaluate
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            # move tensors to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["shape"], batch["color"], batch["sym"])
            loss = criterion(logits, batch["label"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["label"].size(0)
            preds = logits.argmax(-1).detach().cpu().tolist()
            ys.extend(batch["label"].cpu().tolist())
            ps.extend(preds)
            seqs.extend(batch["raw"])
    swa = shape_weighted_accuracy(seqs, ys, ps)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    return tot_loss / max(len(loader.dataset), 1), swa, cwa, ys, ps


# -------------------------------------------------- #
# 8.  Prepare hold-out loader (shared)
holdout_data = load_spr(holdout_name, splits=("test",))
holdout_dl = DataLoader(SPRTorch(holdout_data["test"]), batch_size=64)

# -------------------------------------------------- #
# 9.  Experiment store
experiment_data = {}

# -------------------------------------------------- #
# 10.  Main training loop over variants
for ds_name, _, _ in variants:
    print(f"\n=== Processing dataset: {ds_name} ===")
    experiment_data[ds_name] = {
        "metrics": {"train": [], "val": [], "zrgs": []},
        "losses": {"train": [], "val": []},
        "predictions": {"self_test": [], "holdout_test": []},
        "ground_truth": {"self_test": [], "holdout_test": []},
    }

    spr = load_spr(ds_name)
    train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=64, shuffle=True)
    val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=64)
    test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=64)

    model = DisentangledTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_swa, patience, max_pat = 0.0, 0, 4
    for epoch in range(1, 16):
        tic = time.time()
        tr_loss, tr_swa, tr_cwa, *_ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_swa, val_cwa, *_ = run_epoch(model, val_dl, criterion)
        scheduler.step(val_loss)

        # Zero-shot score on holdout (SWA+CWA)/2
        _, h_swa, h_cwa, *_ = run_epoch(model, holdout_dl, criterion)
        zrg_score = (h_swa + h_cwa) / 2.0

        # record
        experiment_data[ds_name]["losses"]["train"].append((epoch, tr_loss))
        experiment_data[ds_name]["losses"]["val"].append((epoch, val_loss))
        experiment_data[ds_name]["metrics"]["train"].append((epoch, tr_swa, tr_cwa))
        experiment_data[ds_name]["metrics"]["val"].append((epoch, val_swa, val_cwa))
        experiment_data[ds_name]["metrics"]["zrgs"].append((epoch, zrg_score))

        print(
            f"Epoch {epoch:02d} | tr_loss={tr_loss:.3f} val_loss={val_loss:.3f} "
            f"val_SWA={val_swa:.3f} val_CWA={val_cwa:.3f} ZRGS={zrg_score:.3f} "
            f"time={time.time()-tic:.1f}s"
        )

        # early stopping based on SWA
        if val_swa > best_val_swa + 1e-4:
            best_val_swa, patience = val_swa, 0
            torch.save(model.state_dict(), os.path.join(working_dir, f"{ds_name}.pt"))
        else:
            patience += 1
            if patience >= max_pat:
                print("Early stopping triggered.")
                break

    # reload best and final evaluation
    model.load_state_dict(torch.load(os.path.join(working_dir, f"{ds_name}.pt")))
    _, swa_self, cwa_self, gts_self, preds_self = run_epoch(model, test_dl, criterion)
    _, swa_hold, cwa_hold, gts_hold, preds_hold = run_epoch(
        model, holdout_dl, criterion
    )

    experiment_data[ds_name]["predictions"]["self_test"] = preds_self
    experiment_data[ds_name]["predictions"]["holdout_test"] = preds_hold
    experiment_data[ds_name]["ground_truth"]["self_test"] = gts_self
    experiment_data[ds_name]["ground_truth"]["holdout_test"] = gts_hold

    print(
        f"Finished {ds_name}: SWA self={swa_self:.4f} holdout={swa_hold:.4f} "
        f"CWA self={cwa_self:.4f} holdout={cwa_hold:.4f}"
    )

# -------------------------------------------------- #
# 11.  Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "Saved all metrics & predictions to",
    os.path.join(working_dir, "experiment_data.npy"),
)
