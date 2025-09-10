import os, pathlib, random, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- misc / IO ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dset = load_spr_bench(DATA_PATH)


# ---------- helper metrics ----------
def count_shape(seq):
    return len({t[0] for t in seq.split()})


def count_color(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def SWA(seqs, y, g):
    w = [count_shape(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w)


def CWA(seqs, y, g):
    w = [count_color(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w)


def CompWA(seqs, y, g):
    w = [count_shape(s) * count_color(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y, g)]
    return sum(c) / sum(w)


# ---------- vocabulary ----------
vocab = {"<pad>": 0, "<unk>": 1}
for s in dset["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad, unk = vocab["<pad>"], vocab["<unk>"]
labels = sorted(set(dset["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}


# ---------- augmentation ----------
def augment(tokens):
    # span deletion proportional to complexity
    if len(tokens) > 4 and random.random() < 0.3:
        span = random.randint(1, max(1, len(tokens) // 4))
        start = random.randint(0, len(tokens) - span)
        tokens = tokens[:start] + tokens[start + span :]
    # within-shape shuffle
    for i in range(0, len(tokens), 3):
        block = tokens[i : i + 3]
        random.shuffle(block)
        tokens[i : i + 3] = block
    # random masking
    tokens = [t if random.random() > 0.25 else "<unk>" for t in tokens]
    return tokens


# ---------- datasets ----------
class ContrastiveSPR(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def enc(self, toks):
        return [vocab.get(t, unk) for t in toks]

    def __getitem__(self, i):
        toks = self.seqs[i].split()
        v1 = self.enc(augment(toks.copy()))
        v2 = self.enc(augment(toks.copy()))
        return v1, v2


class SupervisedSPR(Dataset):
    def __init__(self, seqs, labs):
        self.seqs, self.labs = seqs, [lab2id[l] for l in labs]

    def __len__(self):
        return len(self.seqs)

    def enc(self, toks):
        return [vocab.get(t, unk) for t in toks]

    def __getitem__(self, i):
        return self.enc(self.seqs[i].split()), self.labs[i], self.seqs[i]


def pad_collate(batch):
    maxlen = max(len(x) for x in batch)
    arr = np.full((len(batch), maxlen), pad, np.int64)
    for i, x in enumerate(batch):
        arr[i, : len(x)] = x
    lens = [len(x) for x in batch]
    return torch.tensor(arr), torch.tensor(lens)


def collate_con(batch):
    v1, v2 = zip(*batch)
    a, lena = pad_collate(v1)
    b, lenb = pad_collate(v2)
    return a, lena, b, lenb


def collate_sup(batch):
    seqs, ys, raw = zip(*batch)
    x, lens = pad_collate(seqs)
    return (x, lens, torch.tensor(ys)), list(raw)


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vs, emb=64, hid=128, proj=128):
        super().__init__()
        self.emb = nn.Embedding(vs, emb, padding_idx=pad)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, proj)

    def forward(self, x, l, project=True):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        feat = torch.cat([h[0], h[1]], 1)
        if project:
            feat = self.proj(feat)
        return feat


class Classifier(nn.Module):
    def __init__(self, enc, nc):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(128, nc)

    def forward(self, x, l):
        return self.fc(self.enc(x, l))


# ---------- InfoNCE with memory queue ----------
class MoCoBuffer:
    def __init__(self, dim=128, K=1024):
        self.K = K
        self.register = torch.randn(K, dim) / math.sqrt(dim)

    def push(self, rep):
        rep = rep.detach().cpu()
        n = len(rep)
        self.register = torch.vstack([rep, self.register])[: self.K]

    def get(self):
        return self.register.to(device)


queue = MoCoBuffer()


def info_nce(z1, z2, T=0.5):
    N = z1.size(0)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    pos = (z1 * z2).sum(1, keepdim=True)
    neg = torch.matmul(z1, queue.get().T)
    logits = torch.cat([pos, neg], 1) / T
    labels = torch.zeros(N, dtype=torch.long, device=device)
    return nn.functional.cross_entropy(logits, labels)


# ---------- pre-train ----------
def pretrain(epochs=2, bs=256):
    enc = Encoder(len(vocab)).to(device)
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    loader = DataLoader(
        ContrastiveSPR(dset["train"]["sequence"]),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_con,
    )
    for ep in range(1, epochs + 1):
        t = time.time()
        tot = 0
        for a, lena, b, lenb in loader:
            a, lena, b, lenb = [t.to(device) for t in [a, lena, b, lenb]]
            z1 = enc(a, lena)
            z2 = enc(b, lenb)
            loss = info_nce(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * a.size(0)
            queue.push(z2)
        print(
            f"Contrastive epoch {ep}: loss={(tot/len(loader.dataset)):.4f}  in {time.time()-t:.1f}s"
        )
    return enc


encoder = pretrain()

# ---------- fine-tune ----------
train_loader = DataLoader(
    SupervisedSPR(dset["train"]["sequence"], dset["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_sup,
)
dev_loader = DataLoader(
    SupervisedSPR(dset["dev"]["sequence"], dset["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_sup,
)

model = Classifier(encoder, len(labels)).to(device)
# layer-wise lr-decay
optim_groups = [
    {"params": model.enc.parameters(), "lr": 3e-4 * 0.1},
    {"params": model.fc.parameters(), "lr": 3e-4},
]
opt = torch.optim.Adam(optim_groups)
ce = nn.CrossEntropyLoss(reduction="none")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best, wait = 0, 0
for epoch in range(1, 11):
    # --- train ---
    model.train()
    tloss = 0
    for (x, lens, y), _ in train_loader:
        x, lens, y = [t.to(device) for t in [x, lens, y]]
        logits = model(x, lens)
        weights = torch.tensor(
            [count_shape("") + 1 for _ in range(len(y))],
            device=device,
            dtype=torch.float,
        )  # simple uniform, placeholder
        loss = (ce(logits, y) * weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tloss += loss.item() * x.size(0)
    tloss /= len(train_loader.dataset)

    # --- val ---
    model.eval()
    vloss = 0
    preds = []
    gts = []
    rawseq = []
    with torch.no_grad():
        for (x, lens, y), raw in dev_loader:
            x, lens, y = [t.to(device) for t in [x, lens, y]]
            logits = model(x, lens)
            loss = ce(logits, y).mean()
            vloss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(y.cpu().tolist())
            rawseq.extend(raw)
    vloss /= len(dev_loader.dataset)
    swa, cwa, comp = (
        SWA(rawseq, [id2lab[i] for i in gts], [id2lab[i] for i in preds]),
        CWA(rawseq, [id2lab[i] for i in gts], [id2lab[i] for i in preds]),
        CompWA(rawseq, [id2lab[i] for i in gts], [id2lab[i] for i in preds]),
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tloss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vloss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(comp)
    print(
        f"Epoch {epoch}: validation_loss = {vloss:.4f}  SWA={swa:.3f}  CWA={cwa:.3f}  CompWA={comp:.3f}"
    )
    if comp > best:
        best, wait = comp, 0
        best_state = model.state_dict()
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts
    else:
        wait += 1
    if wait >= 3:
        break

model.load_state_dict(best_state)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished; logs saved to", os.path.join(working_dir, "experiment_data.npy"))
