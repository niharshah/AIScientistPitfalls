# Mean-Pooling ablation for SPR-BENCH
import os, pathlib, random, time, math, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

warnings.filterwarnings("ignore")

# ---------------- paths / dirs ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- dataset ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adapt if needed
spr = load_spr_bench(DATA_PATH)


# ---------------- metrics helpers -------------
def count_shape_variety(seq):  # first char per token
    return len(set(tok[0] for tok in seq.strip().split()))


def count_color_variety(seq):  # second char per token
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def comp_weight(seq):  # multiplicative weight
    return count_shape_variety(seq) * count_color_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ---------------- vocab / label ----------------
vocab = {"<pad>": 0, "<unk>": 1}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id, unk_id = vocab["<pad>"], vocab["<unk>"]

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)
print(f"Vocab size: {len(vocab)}, #classes: {num_classes}")


# ---------------- augmentation ----------------
def augment(tokens):
    # 30% masking with <unk>
    tokens = [t if random.random() > 0.3 else "<unk>" for t in tokens]
    # shuffle within length-3 windows
    for i in range(0, len(tokens), 3):
        window = tokens[i : i + 3]
        random.shuffle(window)
        tokens[i : i + 3] = window
    return tokens


# ---------------- datasets --------------------
class SPRDatasetPretrain(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        return self.encode(augment(toks.copy())), self.encode(augment(toks.copy()))


class SPRDatasetCLS(Dataset):
    def __init__(self, seqs, labels_):
        self.seqs = seqs
        self.labels = [lab2id[l] for l in labels_]

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.encode(self.seqs[idx].split()), self.labels[idx], self.seqs[idx]


def _pad(seqs):
    lens = [len(s) for s in seqs]
    mx = max(lens)
    arr = np.full((len(seqs), mx), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return torch.tensor(arr), torch.tensor(lens)


def collate_pretrain(batch):
    v1, v2 = zip(*batch)
    a, lena = _pad(v1)
    b, lenb = _pad(v2)
    return a, lena, b, lenb


def collate_cls(batch):
    seqs, ys, raw = zip(*batch)
    x, lens = _pad(seqs)
    return (x, lens, torch.tensor(ys)), list(raw)


# ---------------- encoder (mean-pool) ---------
class MeanPoolingEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_id)
        self.proj = nn.Linear(emb_dim, 128)

    def forward(self, x, lens, project=True):
        e = self.emb(x)  # B,T,D
        mask = (x != pad_id).unsqueeze(-1)  # B,T,1
        summed = (e * mask).sum(1)  # B,D
        mean = summed / lens.unsqueeze(-1)  # B,D
        return self.proj(mean) if project else mean  # B,128 or B,D


class Classifier(nn.Module):
    def __init__(self, encoder, nclass):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(128, nclass)

    def forward(self, x, lens):
        z = self.enc(x, lens, project=True)
        return self.head(z)


# ---------------- contrastive loss ------------
def nt_xent(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    N = z1.size(0)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss = 0.5 * (
        nn.functional.cross_entropy(sim[:N], targets)
        + nn.functional.cross_entropy(sim[N:], targets - N)
    )
    return loss


# ---------------- pretrain --------------------
def pretrain_encoder(epochs=3, batch=256, lr=1e-3):
    enc = MeanPoolingEncoder(len(vocab)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    loader = DataLoader(
        SPRDatasetPretrain(spr["train"]["sequence"]),
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_pretrain,
    )
    enc.train()
    for ep in range(1, epochs + 1):
        t0, tot = time.time(), 0
        for a, lena, b, lenb in loader:
            a, lena, b, lenb = (
                a.to(device),
                lena.to(device),
                b.to(device),
                lenb.to(device),
            )
            z1, z2 = enc(a, lena), enc(b, lenb)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * a.size(0)
        print(
            f"Pre-train epoch {ep}: loss={tot/len(loader.dataset):.4f}  ({time.time()-t0:.1f}s)"
        )
    return enc


pretrained_enc = pretrain_encoder()

# ---------------- fine-tuning -----------------
train_loader = DataLoader(
    SPRDatasetCLS(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    SPRDatasetCLS(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_cls,
)

model = Classifier(pretrained_enc, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "MeanPooling": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

best_val, wait, patience = 1e9, 0, 3
for epoch in range(1, 21):
    # ---- train ----
    model.train()
    tr_loss = 0
    for (x, lens, y), _ in train_loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_loader.dataset)
    # ---- val ----
    model.eval()
    val_loss = 0
    preds = []
    gt = []
    rawseq = []
    with torch.no_grad():
        for (x, lens, y), raw in dev_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            out = model(x, lens)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gt.extend(y.cpu().tolist())
            rawseq.extend(raw)
    val_loss /= len(dev_loader.dataset)
    cwa = comp_weighted_accuracy(
        rawseq, [id2lab[i] for i in gt], [id2lab[i] for i in preds]
    )
    # log
    exp = experiment_data["MeanPooling"]["SPR_BENCH"]
    exp["losses"]["train"].append(tr_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["val"].append(cwa)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}  CompWA={cwa:.4f}")
    # early stop
    if val_loss < best_val - 1e-4:
        best_val, wait = val_loss, 0
        exp["predictions"] = preds
        exp["ground_truth"] = gt
        best_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        break

model.load_state_dict(best_state)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Saved to working/experiment_data.npy")
