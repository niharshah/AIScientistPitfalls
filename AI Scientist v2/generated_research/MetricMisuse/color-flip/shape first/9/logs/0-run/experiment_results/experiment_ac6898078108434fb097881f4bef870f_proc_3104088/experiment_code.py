import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict as HFDD

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- helper: data & metrics ----------
def load_spr_bench(root: pathlib.Path):
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = HFDD()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


# ---------- dataset wrappers ----------
class SPRContrastive(Dataset):
    def __init__(self, seqs, vocab):
        self.seqs, self.vocab = seqs, vocab

    def __len__(self):
        return len(self.seqs)

    def encode(self, toks):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in toks]

    def augment(self, seq):
        toks = seq.split()
        view = [t for t in toks if random.random() > 0.2]  # 20% drop
        if len(view) == 0:
            view = [random.choice(toks)]
        return self.encode(view)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        return self.augment(s), self.augment(s)


class SPRLabelled(Dataset):
    def __init__(self, seqs, labels, vocab, lbl2id):
        self.seqs, self.labels, self.vocab, self.lbl2id = seqs, labels, vocab, lbl2id

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]

    def __getitem__(self, idx):
        return (
            self.encode(self.seqs[idx]),
            self.lbl2id[self.labels[idx]],
            self.seqs[idx],
        )


def pad_collate(batch):
    if isinstance(batch[0][0], list) and isinstance(batch[0][1], list):  # contrastive
        v1, v2 = zip(*batch)
        return _pad(v1), _pad(v2)
    seqs, labels, raw = zip(*batch)
    return _pad(seqs), torch.tensor(labels), raw


def _pad(seq_lists):
    lens = [len(s) for s in seq_lists]
    maxlen = max(lens)
    arr = np.full((len(seq_lists), maxlen), fill_value=0, dtype=np.int64)
    for i, seq in enumerate(seq_lists):
        arr[i, : len(seq)] = seq
    return torch.tensor(arr)


# ---------- vocab & label maps ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)

vocab = {"<pad>": 0, "<unk>": 1}
for s in dsets["train"]["sequence"]:
    for t in s.split():
        if t not in vocab:
            vocab[t] = len(vocab)
labels_sorted = sorted(set(dsets["train"]["label"]))
lbl2id = {l: i for i, l in enumerate(labels_sorted)}
id2lbl = {i: l for l, i in lbl2id.items()}


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        z = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return z


class ProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    def __init__(self, enc, emb_dim, n_cls):
        super().__init__()
        self.enc = enc
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, x):
        return self.fc(self.enc(x))


# ---------- contrastive loss ----------
def nt_xent(z1, z2, T=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / T
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


# ---------- pre-training ----------
enc = Encoder(len(vocab), 64).to(device)
proj = ProjectionHead(64, 64).to(device)
opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=3e-4)

pre_loader = DataLoader(
    SPRContrastive(dsets["train"]["sequence"], vocab),
    batch_size=256,
    shuffle=True,
    collate_fn=pad_collate,
)
pre_epochs = 5
print("Start contrastive pre-training")
for ep in range(1, pre_epochs + 1):
    enc.train()
    proj.train()
    tot_loss = 0
    st = time.time()
    for v1, v2 in pre_loader:
        v1, v2 = v1.to(device), v2.to(device)
        z1, z2 = proj(enc(v1)), proj(enc(v2))
        loss = nt_xent(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_loss += loss.item() * v1.size(0)
    print(
        f"Pre-train Epoch {ep}/{pre_epochs} loss {tot_loss/len(pre_loader.dataset):.4f}  ({time.time()-st:.1f}s)"
    )

# ---------- fine-tuning ----------
train_loader = DataLoader(
    SPRLabelled(dsets["train"]["sequence"], dsets["train"]["label"], vocab, lbl2id),
    batch_size=128,
    shuffle=True,
    collate_fn=pad_collate,
)
dev_loader = DataLoader(
    SPRLabelled(dsets["dev"]["sequence"], dsets["dev"]["label"], vocab, lbl2id),
    batch_size=256,
    shuffle=False,
    collate_fn=pad_collate,
)

clf = Classifier(enc, 64, len(labels_sorted)).to(device)  # encoder reused!
criterion = nn.CrossEntropyLoss()
opt2 = torch.optim.Adam(clf.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

patience, wait, max_epochs = 3, 0, 20
best_val = -1
best_state = None
for ep in range(1, max_epochs + 1):
    # ---- train ----
    clf.train()
    tr_loss = 0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(clf(x), y)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_loader.dataset)
    # ---- val ----
    clf.eval()
    val_loss, preds, gts, raw_seqs = 0, [], [], []
    with torch.no_grad():
        for x, y, r in dev_loader:
            x, y = x.to(device), y.to(device)
            out = clf(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(y.cpu().tolist())
            raw_seqs.extend(r)
    val_loss /= len(dev_loader.dataset)
    compWA = comp_weighted_accuracy(
        raw_seqs, [id2lbl[i] for i in gts], [id2lbl[i] for i in preds]
    )
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f}, CompWA = {compWA:.4f}")
    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(compWA)
    # early stopping
    if compWA > best_val:
        best_val, best_state, wait = compWA, clf.state_dict(), 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping")
        break

# restore best model
clf.load_state_dict(best_state)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Best CompWA achieved on dev: {best_val:.4f}")
