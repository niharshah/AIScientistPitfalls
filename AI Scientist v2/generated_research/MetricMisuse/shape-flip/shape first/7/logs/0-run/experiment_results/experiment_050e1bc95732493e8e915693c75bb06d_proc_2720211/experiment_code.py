import os, random, torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------
# Working dir / device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------
# Experiment-wide store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------
# Metrics
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


# -----------------------------------------------------------
# Data loading helpers
def load_spr_bench(root: str) -> DatasetDict:
    def _ld(split):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{split}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train"), dev=_ld("dev"), test=_ld("test"))


def make_toy_set(path: str, n_tr=5000, n_dev=1000, n_te=1000):
    shapes, colors = list("STCH"), list("RGBY")

    def rnd_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def lbl(seq):  # simple rule: 1 if #shape types is even
        return int(count_shape_variety(seq) % 2 == 0)

    os.makedirs(path, exist_ok=True)
    for n, split in [(n_tr, "train"), (n_dev, "dev"), (n_te, "test")]:
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rnd_seq()
                f.write(f"{i},{s},{lbl(s)}\n")


root = "SPR_BENCH"
if not (
    os.path.isdir(root)
    and all(
        os.path.isfile(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found – generating toy data.")
    make_toy_set(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Vocabulary
def build_vocab(hf_ds):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in hf_ds["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(tok, 1) for tok in seq.split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------
# Torch Dataset
class SPRTorch(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]
        self.labels = hf["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        sym = np.array(
            [count_shape_variety(seq), len(set(tok[1] for tok in seq.split()))],
            dtype=np.float32,
        )
        return {
            "input": torch.tensor(encode(seq), dtype=torch.long),
            "sym": torch.tensor(sym),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


bs = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=bs, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=bs)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=bs)


# -----------------------------------------------------------
# Model
class GateTransformer(nn.Module):
    def __init__(
        self, vocab_sz, emb=128, heads=4, layers=2, sym_dim=2, cls=2, use_symbolic=True
    ):
        super().__init__()
        self.use_symbolic = use_symbolic
        self.embed = nn.Embedding(vocab_sz, emb, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.sym_mlp = nn.Linear(sym_dim, emb)
        self.gate_net = nn.Sequential(nn.Linear(sym_dim, emb), nn.Sigmoid())
        self.cls = nn.Linear(emb, cls)

    def forward(self, tok, sym):
        x = self.embed(tok)  # (B,L,emb)
        h = self.encoder(x).mean(1)  # (B,emb)
        if self.use_symbolic:
            sym_emb = torch.relu(self.sym_mlp(sym))
            gate = self.gate_net(sym)  # (B,emb) in [0,1]
            h = gate * h + (1 - gate) * sym_emb
        return self.cls(h)


use_symbolic = True  # toggle off for ablation
model = GateTransformer(len(vocab), use_symbolic=use_symbolic).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)


# -----------------------------------------------------------
# Epoch helpers
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    tot_loss = 0
    yt, yp, seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["input"], batch["sym"])
            loss = criterion(out, batch["label"])
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            tot_loss += loss.item() * batch["label"].size(0)
            preds = out.argmax(-1).detach().cpu().tolist()
            yp.extend(preds)
            yt.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw"])
    return tot_loss / len(dl.dataset), shape_weighted_accuracy(seqs, yt, yp), yt, yp


# -----------------------------------------------------------
epochs = 6
for ep in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, train=True)
    v_loss, v_swa, _, _ = run_epoch(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, v_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((ep, tr_swa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, v_swa))
    print(f"Epoch {ep}: validation_loss = {v_loss:.4f}, SWA = {v_swa:.4f}")

# -----------------------------------------------------------
# Test
_, test_swa, gt, pred = run_epoch(test_dl)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = pred
experiment_data["SPR_BENCH"]["ground_truth"] = gt

# -----------------------------------------------------------
# Save artefacts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
