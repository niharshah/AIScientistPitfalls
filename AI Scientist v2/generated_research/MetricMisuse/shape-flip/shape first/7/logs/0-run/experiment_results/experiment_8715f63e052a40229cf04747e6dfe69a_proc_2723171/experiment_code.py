import os, random, torch, numpy as np, pathlib, time
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- housekeeping ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
disable_caching()  # keep HF cache clean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {"MultiSynGen": {}}  # storage for all metrics / predictions


# ---------- helper metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


def zero_shot_rule_gen_score(seqs, y_true, y_pred, seen_pairs):
    """ZRGS = mean(SWA, CWA) on examples whose (shapeVar, colorVar) never appears in training set"""
    idx_unseen = [
        i
        for i, s in enumerate(seqs)
        if (count_shape_variety(s), count_color_variety(s)) not in seen_pairs
    ]
    if not idx_unseen:
        return 0.0
    u_seqs = [seqs[i] for i in idx_unseen]
    u_true = [y_true[i] for i in idx_unseen]
    u_pred = [y_pred[i] for i in idx_unseen]
    swa, cwa = shape_weighted_accuracy(u_seqs, u_true, u_pred), color_weighted_accuracy(
        u_seqs, u_true, u_pred
    )
    return 0.5 * (swa + cwa)


# ---------- synthetic-data generation ----------
shapes, colors = list("STCH"), list("RGBY")


def rnd_seq():
    L = random.randint(3, 9)
    return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))


def make_dataset(path: str, rule_fn, seed: int, n_tr=6000, n_dev=1500, n_te=1500):
    if os.path.isdir(path) and any(os.scandir(path)):
        return  # already there
    print("Creating synthetic data at", path)
    os.makedirs(path, exist_ok=True)
    random.seed(seed)
    for n, split in [(n_tr, "train"), (n_dev, "dev"), (n_te, "test")]:
        if n == 0:  # skip empty split
            continue
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = rnd_seq()
                sh = len(set(t[0] for t in seq.split()))
                co = len(set(t[1] for t in seq.split()))
                f.write(f"{i},{seq},{rule_fn(sh, co)}\n")


def load_spr(root: str) -> DatasetDict:
    root = pathlib.Path(root)
    d = {}
    for split in ["train", "dev", "test"]:
        csv_path = root / f"{split}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            d[split] = load_dataset(
                "csv", data_files=str(csv_path), split="train", cache_dir=".cache_dsets"
            )
    return DatasetDict(d)


# ---------- dataset variants ----------
variants = [
    ("delta0_gt", lambda sh, co: int(sh > co), 11),  # shapes  > colours
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
make_dataset(
    holdout_name, holdout_rule, holdout_seed, n_tr=0, n_dev=0, n_te=2000
)  # only test split

# ---------- vocab / encoding ----------
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


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

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


# ---------- model ----------
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
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_cls)

    def forward(self, sh_ids, col_ids, sym_feats):
        tok_emb = self.shape_emb(sh_ids) + self.color_emb(col_ids)
        enc_out = self.encoder(tok_emb)
        pooled = enc_out.mean(1)
        sym_emb = torch.relu(self.sym_fc(sym_feats))
        logits = self.out(torch.cat([pooled, sym_emb], -1))
        return logits


# ---------- train / eval ----------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            # move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
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
    avg_loss = tot_loss / max(len(loader.dataset), 1)
    swa = shape_weighted_accuracy(seqs, ys, ps)
    cwa = color_weighted_accuracy(seqs, ys, ps)
    return avg_loss, swa, cwa, ys, ps, seqs


# ---------- shared hold-out loader ----------
holdout_data = load_spr(holdout_name)
holdout_dl = DataLoader(SPRTorch(holdout_data["test"]), batch_size=64)

# ---------- main training loop ----------
for ds_name, ds_rule, _ in variants:
    print(f"\n=== Processing dataset: {ds_name} ===")
    experiment_data["MultiSynGen"][ds_name] = {
        "metrics": {"train": [], "val": [], "self_test": None, "holdout_test": None},
        "losses": {"train": [], "val": []},
        "predictions": {"self_test": [], "holdout_test": []},
        "ground_truth": {"self_test": [], "holdout_test": []},
    }

    # load data
    spr = load_spr(ds_name)
    train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=64, shuffle=True)
    val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=64)
    test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=64)

    # pre-compute seen (shapeVar, colorVar) pairs in training
    seen_pairs = set(
        (count_shape_variety(s), count_color_variety(s))
        for s in spr["train"]["sequence"]
    )

    # model, criterions, optimiser
    model = DisentangledTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val, patience, max_pat = 0.0, 0, 4
    for epoch in range(1, 16):
        start_ep = time.time()
        tr_loss, tr_swa, tr_cwa, _, _, _ = run_epoch(
            model, train_dl, criterion, optimizer
        )
        val_loss, val_swa, val_cwa, ys, ps, seqs = run_epoch(model, val_dl, criterion)

        val_zrgs = zero_shot_rule_gen_score(seqs, ys, ps, seen_pairs)
        scheduler.step(val_loss)

        experiment_data["MultiSynGen"][ds_name]["losses"]["train"].append(
            (epoch, tr_loss)
        )
        experiment_data["MultiSynGen"][ds_name]["losses"]["val"].append(
            (epoch, val_loss)
        )
        experiment_data["MultiSynGen"][ds_name]["metrics"]["train"].append(
            (epoch, tr_swa, tr_cwa, 0.0)
        )
        experiment_data["MultiSynGen"][ds_name]["metrics"]["val"].append(
            (epoch, val_swa, val_cwa, val_zrgs)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
            f"SWA_val={val_swa:.4f} CWA_val={val_cwa:.4f} ZRGS_val={val_zrgs:.4f} "
            f"time={time.time()-start_ep:.1f}s"
        )

        if val_swa > best_val + 1e-4:
            best_val, patience = val_swa, 0
            torch.save(model.state_dict(), os.path.join(working_dir, f"{ds_name}.pt"))
        else:
            patience += 1
            if patience >= max_pat:
                print("Early stopping.")
                break

    # -------- evaluation on best checkpoint --------
    model.load_state_dict(torch.load(os.path.join(working_dir, f"{ds_name}.pt")))
    _, swa_self, cwa_self, gts_self, preds_self, seqs_self = run_epoch(
        model, test_dl, criterion
    )
    zrg_self = zero_shot_rule_gen_score(seqs_self, gts_self, preds_self, seen_pairs)

    _, swa_hold, cwa_hold, gts_hold, preds_hold, seqs_hold = run_epoch(
        model, holdout_dl, criterion
    )
    # for hold-out, seen_pairs use same training seen_pairs to measure true zero-shot
    zrg_hold = zero_shot_rule_gen_score(seqs_hold, gts_hold, preds_hold, seen_pairs)

    experiment_data["MultiSynGen"][ds_name]["metrics"]["self_test"] = (
        swa_self,
        cwa_self,
        zrg_self,
    )
    experiment_data["MultiSynGen"][ds_name]["metrics"]["holdout_test"] = (
        swa_hold,
        cwa_hold,
        zrg_hold,
    )
    experiment_data["MultiSynGen"][ds_name]["predictions"]["self_test"] = preds_self
    experiment_data["MultiSynGen"][ds_name]["predictions"]["holdout_test"] = preds_hold
    experiment_data["MultiSynGen"][ds_name]["ground_truth"]["self_test"] = gts_self
    experiment_data["MultiSynGen"][ds_name]["ground_truth"]["holdout_test"] = gts_hold

    print(
        f"Finished {ds_name}: "
        f"SWA self={swa_self:.4f}, holdout={swa_hold:.4f} | "
        f"ZRGS holdout={zrg_hold:.4f}"
    )

# ---------- save everything ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
