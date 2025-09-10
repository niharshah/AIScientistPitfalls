import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------------- paths / dirs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------------- device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------------- reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# --------------------------------------------------------------------------------- metric helpers
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) + 1e-9)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) + 1e-9)


# --------------------------------------------------------------------------------- data loading
def load_real_spr(root: pathlib.Path):
    """Return splits as python dicts (columns -> list).  FIXED .to_dict() bug"""
    from datasets import load_dataset

    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    splits = {name: _ld(f"{name}.csv") for name in ("train", "dev", "test")}
    # convert each HF Dataset to pure python dict (column -> list)
    return {name: splits[name].to_dict() for name in splits}


def make_synth(n: int):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labs = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labs.append(int(count_shape_variety(seq) >= count_color_variety(seq)))
    return {"sequence": seqs, "label": labs}


SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if SPR_PATH.exists():
    try:
        raw = load_real_spr(SPR_PATH)
        print("Loaded real SPR_BENCH dataset")
    except Exception as e:
        print("Failed loading real SPR, using synthetic instead.", e)
        raw = {}
else:
    raw = {}
if not raw:
    raw = {"train": make_synth(4000), "dev": make_synth(800), "test": make_synth(800)}
    print("Synthetic dataset created")

# --------------------------------------------------------------------------------- vocab & encoding
special = ["<pad>", "<unk>", "<mask>"]
tokens = set(tok for seq in raw["train"]["sequence"] for tok in seq.split())
vocab = {tok: i + len(special) for i, tok in enumerate(sorted(tokens))}
for i, t in enumerate(special):
    vocab[t] = i
pad_idx, unk_idx, mask_idx = vocab["<pad>"], vocab["<unk>"], vocab["<mask>"]
max_len = max(len(seq.split()) for seq in raw["train"]["sequence"])


def encode(seq: str):
    idxs = [vocab.get(t, unk_idx) for t in seq.split()]
    idxs = idxs[:max_len] + [pad_idx] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


# --------------------------------------------------------------------------------- augmentation
def augment(seq, mask_p=0.15, shuffle_p=0.30):
    toks = seq.split()
    # mask
    for i in range(len(toks)):
        if random.random() < mask_p:
            toks[i] = "<mask>"
    # shuffle contiguous span
    if random.random() < shuffle_p and len(toks) > 2:
        i, j = sorted(random.sample(range(len(toks)), 2))
        toks[i:j] = random.sample(toks[i:j], len(toks[i:j]))
    return " ".join(toks)


# --------------------------------------------------------------------------------- datasets
class SPRClsDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx]),
            "raw": self.seqs[idx],
        }


class SPRContrastiveDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        return {"x1": encode(augment(s)), "x2": encode(augment(s))}


train_contrast_ds = SPRContrastiveDataset(raw["train"]["sequence"])
train_cls_ds = SPRClsDataset(raw["train"]["sequence"], raw["train"]["label"])
dev_cls_ds = SPRClsDataset(raw["dev"]["sequence"], raw["dev"]["label"])
test_cls_ds = SPRClsDataset(raw["test"]["sequence"], raw["test"]["label"])

contr_loader = DataLoader(train_contrast_ds, batch_size=256, shuffle=True)
train_loader = DataLoader(train_cls_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_cls_ds, batch_size=512)
test_loader = DataLoader(test_cls_ds, batch_size=512)


# --------------------------------------------------------------------------------- models
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.to(device)
        rep, _ = self.rnn(self.emb(x))
        rep = rep.mean(1)
        return nn.functional.normalize(rep, dim=1)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.fc(z)


def nt_xent(z1, z2, temp=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.mm(z, z.T) / temp
    sim_exp = torch.exp(sim - torch.max(sim, 1, keepdim=True)[0])
    mask = (~torch.eye(2 * N, dtype=torch.bool, device=z1.device)).float()
    sim_exp = sim_exp * mask
    denom = sim_exp.sum(1, keepdim=True)
    pos = torch.exp((z1 * z2).sum(1) / temp)
    loss = -torch.log(pos / denom[:N, 0]) - torch.log(pos / denom[N:, 0])
    return loss.mean()


# --------------------------------------------------------------------------------- bookkeeping
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "ACA": {"val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# --------------------------------------------------------------------------------- pre-training
enc = Encoder(len(vocab)).to(device)
opt_enc = torch.optim.Adam(enc.parameters(), lr=1e-3)
pre_epochs = 3
print("\nPre-training encoder (contrastive)…")
for ep in range(1, pre_epochs + 1):
    enc.train()
    tot, n = 0, 0
    for batch in contr_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z1, z2 = enc(batch["x1"]), enc(batch["x2"])
        loss = nt_xent(z1, z2)
        opt_enc.zero_grad()
        loss.backward()
        opt_enc.step()
        tot += loss.item() * z1.size(0)
        n += z1.size(0)
    print(f"  Contrastive epoch {ep}: loss={tot/n:.4f}")

# --------------------------------------------------------------------------------- fine-tuning
model = Classifier(enc, num_classes=len(set(raw["train"]["label"]))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
epochs = 5


def evaluate(loader):
    model.eval()
    preds, gts, raws, tot, n = [], [], [], 0, 0
    with torch.no_grad():
        for batch in loader:
            raws.extend(batch["raw"])
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            tot += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(batch["y"].cpu().tolist())
    return tot / n, preds, gts, raws


def compute_aca(seqs, labels, M=3):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s, l in zip(seqs, labels):
            for _ in range(M):
                aug = encode(augment(s, 0.1, 0.2)).unsqueeze(0).to(device)
                if model(aug).argmax(1).item() == l:
                    correct += 1
    return correct / (len(seqs) * M)


best_val_swa = -1
for ep in range(1, epochs + 1):
    # train
    model.train()
    tot, n = 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        loss = criterion(model(batch["x"]), batch["y"])
        loss.backward()
        optimizer.step()
        tot += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    train_loss = tot / n

    val_loss, val_preds, val_gts, val_raws = evaluate(dev_loader)
    val_swa = shape_weighted_accuracy(val_raws, val_gts, val_preds)
    val_aca = compute_aca(raw["dev"]["sequence"], raw["dev"]["label"], M=3)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    experiment_data["SPR_BENCH"]["ACA"]["val"].append(val_aca)

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={val_swa:.4f} | ACA={val_aca:.4f}"
    )

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {
            k: v.cpu() for k, v in model.state_dict().items()
        }  # save to CPU to avoid GPU ref

# --------------------------------------------------------------------------------- test
model.load_state_dict(best_state)
test_loss, test_preds, test_gts, test_raws = evaluate(test_loader)
test_swa = shape_weighted_accuracy(test_raws, test_gts, test_preds)
test_cwa = color_weighted_accuracy(test_raws, test_gts, test_preds)
test_aca = compute_aca(raw["test"]["sequence"], raw["test"]["label"], M=3)

print(f"\nTest results: SWA={test_swa:.4f} | CWA={test_cwa:.4f} | ACA={test_aca:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test_SWA"] = test_swa
experiment_data["SPR_BENCH"]["metrics"]["test_CWA"] = test_cwa
experiment_data["SPR_BENCH"]["ACA"]["test"] = test_aca
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
