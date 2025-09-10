# REMOVE-COLOR ablation for the SPR benchmark  – single-file script
import os, random, torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------ #
# 0.   House-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------ #
# 1.   Experiment-wide data store
experiment_data = {
    "REMOVE_COLOR": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ------------------------------------------------------------------ #
# 2.   Metric helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


# ------------------------------------------------------------------ #
# 3.   SPR benchmark loader (makes synthetic fallback)
def load_spr_bench(root: str) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{split_name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def make_synthetic_dataset(path: str, n_tr=8000, n_dev=2000, n_te=2000):
    shapes, colors = list("STCH"), list("RGBY")

    def rnd_seq():
        L = random.randint(3, 9)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # label 1 if unique-shapes > unique-colours else 0
        sh = len(set(t[0] for t in seq.split()))
        co = len(set(t[1] for t in seq.split()))
        return int(sh > co)

    os.makedirs(path, exist_ok=True)
    for n, split in [(n_tr, "train"), (n_dev, "dev"), (n_te, "test")]:
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rnd_seq()
                f.write(f"{i},{s},{rule(s)}\n")


root_path = "SPR_BENCH"
if not (
    os.path.isdir(root_path)
    and all(
        os.path.isfile(os.path.join(root_path, f"{s}.csv"))
        for s in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found, creating synthetic fallback.")
    make_synthetic_dataset(root_path)

spr = load_spr_bench(root_path)
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------ #
# 4.   Vocabularies
pad_idx, max_len = 0, 20
shape_vocab = {s: i + 1 for i, s in enumerate(sorted({"S", "T", "C", "H"}))}
color_vocab = {c: i + 1 for i, c in enumerate(sorted({"R", "G", "B", "Y"}))}


def encode(seq):
    s_ids, c_ids = [], []
    for tok in seq.split()[:max_len]:
        s_ids.append(shape_vocab.get(tok[0], 0))
        c_ids.append(color_vocab.get(tok[1], 0) if len(tok) > 1 else 0)
    while len(s_ids) < max_len:
        s_ids.append(pad_idx)
        c_ids.append(pad_idx)
    return s_ids, c_ids


# ------------------------------------------------------------------ #
# 5.   Dataset wrapper
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        s_ids, c_ids = encode(seq)
        sym_feats = torch.tensor(
            [count_shape_variety(seq), len(set(tok[1] for tok in seq.split()))],
            dtype=torch.float,
        )
        return {
            "shape": torch.tensor(s_ids, dtype=torch.long),
            "color": torch.tensor(c_ids, dtype=torch.long),  # kept for API consistency
            "sym": sym_feats,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# ------------------------------------------------------------------ #
# 6.   Transformer WITHOUT colour modality
class NoColorTransformer(nn.Module):
    def __init__(self, shp_vocab, d_model=64, nhead=4, nlayers=2, sym_dim=2, n_cls=2):
        super().__init__()
        self.shape_emb = nn.Embedding(len(shp_vocab) + 1, d_model, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_cls)

    def forward(self, sh_ids, col_ids, sym_feats):  # col_ids ignored
        tok_emb = self.shape_emb(sh_ids)  # NO colour embeddings added
        enc = self.encoder(tok_emb)
        pooled = enc.mean(1)
        sym_emb = torch.relu(self.sym_fc(sym_feats))
        return self.out(torch.cat([pooled, sym_emb], dim=-1))


model = NoColorTransformer(shape_vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2
)


# ------------------------------------------------------------------ #
# 7.   Train / Eval functions
def run_epoch(loader, is_train=False):
    model.train() if is_train else model.eval()
    tot_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
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
            pred = logits.argmax(-1).detach().cpu().tolist()
            ys.extend(batch["label"].cpu().tolist())
            ps.extend(pred)
            seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(loader.dataset)
    swa = shape_weighted_accuracy(seqs, ys, ps)
    return avg_loss, swa, ys, ps


# ------------------------------------------------------------------ #
# 8.   Training loop with early stopping
best_val, patience, max_pat = 0.0, 0, 4
epochs = 15
for ep in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, True)
    val_loss, val_swa, _, _ = run_epoch(val_dl, False)
    scheduler.step(val_loss)
    experiment_data["REMOVE_COLOR"]["SPR_BENCH"]["losses"]["train"].append(
        (ep, tr_loss)
    )
    experiment_data["REMOVE_COLOR"]["SPR_BENCH"]["losses"]["val"].append((ep, val_loss))
    experiment_data["REMOVE_COLOR"]["SPR_BENCH"]["metrics"]["train"].append(
        (ep, tr_swa)
    )
    experiment_data["REMOVE_COLOR"]["SPR_BENCH"]["metrics"]["val"].append((ep, val_swa))
    print(f"Epoch {ep}: val_loss={val_loss:.4f}  SWA={val_swa:.4f}")
    if val_swa > best_val + 1e-4:
        best_val, patience = val_swa, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        patience += 1
        if patience >= max_pat:
            print("Early stopping.")
            break

# ------------------------------------------------------------------ #
# 9.   Test evaluation
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
_, test_swa, gts, preds = run_epoch(test_dl, False)
print(f"Test SWA = {test_swa:.4f}")
exp = experiment_data["REMOVE_COLOR"]["SPR_BENCH"]
exp["predictions"], exp["ground_truth"] = preds, gts

# ------------------------------------------------------------------ #
# 10.  Persist experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
