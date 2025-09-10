# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------- #
# 0.  House-keeping, working dir, device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------- #
# 1.   Experiment-wide store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# --------------------------------------------- #
# 2.   Metrics helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


# --------------------------------------------- #
# 3.   SPR loader (with synthetic fallback)
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

    def rule(seq):  # toy rule: label 1 if unique-shapes > unique-colours else 0
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
    print("SPR_BENCH not found, creating synthetic data.")
    make_synthetic_dataset(root_path)

spr = load_spr_bench(root_path)
print({k: len(v) for k, v in spr.items()})

# --------------------------------------------- #
# 4.   Vocabularies for shapes & colours
shape_vocab = {
    s: i + 1
    for i, s in enumerate(
        sorted({tok[0] for tok in spr["train"]["sequence"][0].split()} | set("STCH"))
    )
}
color_vocab = {
    c: i + 1
    for i, c in enumerate(
        sorted({tok[1] for tok in spr["train"]["sequence"][0].split()} | set("RGBY"))
    )
}

pad_idx = 0
max_len = 20


def encode(seq):
    s_ids, c_ids = [], []
    for tok in seq.split()[:max_len]:
        s_ids.append(shape_vocab.get(tok[0], 0))
        c_ids.append(color_vocab.get(tok[1], 0) if len(tok) > 1 else 0)
    # padding
    while len(s_ids) < max_len:
        s_ids.append(pad_idx)
        c_ids.append(pad_idx)
    return s_ids, c_ids


# --------------------------------------------- #
# 5.   PyTorch Dataset wrapper
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
            [count_shape_variety(seq), len(set(tok[1] for tok in seq.split()))],
            dtype=torch.float,
        )
        return {
            "shape": torch.tensor(s_ids, dtype=torch.long),
            "color": torch.tensor(c_ids, dtype=torch.long),
            "sym": sym_feats,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# --------------------------------------------- #
# 6.   Neural-symbolic Transformer model
class DisentangledTransformer(nn.Module):
    def __init__(
        self, shp_vocab, col_vocab, d_model=64, nhead=4, nlayers=2, sym_dim=2, n_cls=2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(len(shp_vocab) + 1, d_model, padding_idx=pad_idx)
        self.color_emb = nn.Embedding(len(col_vocab) + 1, d_model, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_cls)  # pooled + sym

    def forward(self, sh_ids, col_ids, sym_feats):
        tok_emb = self.shape_emb(sh_ids) + self.color_emb(col_ids)
        enc = self.encoder(tok_emb)
        pooled = enc.mean(1)  # order-invariant
        sym_emb = torch.relu(self.sym_fc(sym_feats))
        logits = self.out(torch.cat([pooled, sym_emb], dim=-1))
        return logits


model = DisentangledTransformer(shape_vocab, color_vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2
)


# --------------------------------------------- #
# 7.   Train / Eval functions
def run_epoch(loader, is_train=False):
    model.train() if is_train else model.eval()
    total_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            # move tensors
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
            total_loss += loss.item() * batch["label"].size(0)
            pred = logits.argmax(-1).detach().cpu().tolist()
            ys.extend(batch["label"].cpu().tolist())
            ps.extend(pred)
            seqs.extend(batch["raw"])
    avg_loss = total_loss / len(loader.dataset)
    swa = shape_weighted_accuracy(seqs, ys, ps)
    return avg_loss, swa, ys, ps


# --------------------------------------------- #
# 8.   Training loop with early stopping
best_val, patience, max_pat = 0.0, 0, 4
epochs = 15
for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, True)
    val_loss, val_swa, _, _ = run_epoch(val_dl, False)
    scheduler.step(val_loss)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((epoch, tr_swa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_swa))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")
    # early stop
    if val_swa > best_val + 1e-4:
        best_val, patience = val_swa, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        patience += 1
        if patience >= max_pat:
            print("Early stopping.")
            break

# --------------------------------------------- #
# 9.   Test set evaluation
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
_, test_swa, gts, preds = run_epoch(test_dl, False)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# --------------------------------------------- #
# 10.   Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Metrics saved to", os.path.join(working_dir, "experiment_data.npy"))
