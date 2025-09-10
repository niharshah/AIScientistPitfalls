import os, pathlib, random, time, math, numpy as np, torch, itertools
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------ experiment bookkeeping ------------------
experiment_data = {
    "MultiDatasetContrastivePretrain": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------ working dir -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ helper: load SPR ------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

train_size = len(spr["train"])

# ------------------ synthetic corpora -----------------------
colors = ["red", "blue", "green", "yellow", "orange", "purple"]
shapes = ["circle", "square", "triangle", "star", "oval", "hex"]
tokens_lp = ["x", "y", "z", "w"]


def gen_color_shape_reversal(n):
    seqs = []
    for _ in range(n):
        L = random.randint(4, 10)
        tks = []
        for _ in range(L):
            c = random.choice(colors)
            s = random.choice(shapes)
            if random.random() < 0.5:
                tks.extend([c, s])
            else:  # reversal
                tks.extend([s, c])
        seqs.append(" ".join(tks))
    return seqs


def gen_length_parity(n):
    seqs = []
    for _ in range(n):
        L = random.randint(2, 20)
        tks = [random.choice(tokens_lp) for __ in range(L)]
        seqs.append(" ".join(tks))
    return seqs


syn1 = gen_color_shape_reversal(train_size)
syn2 = gen_length_parity(train_size)
combined_train_sequences = list(itertools.chain(spr["train"]["sequence"], syn1, syn2))


# ------------------ evaluation helpers ----------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def comp_weight(seq):
    return count_shape_variety(seq) * count_color_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ------------------ vocabulary ------------------------------
vocab = {"<pad>": 0, "<unk>": 1}
for s in combined_train_sequences:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]
labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)
print(f"Vocabulary size = {len(vocab)}, num classes = {num_classes}")


# ------------------ augmentation ----------------------------
def augment(tokens):
    tokens = [t if random.random() > 0.3 else "<unk>" for t in tokens]
    for i in range(0, len(tokens), 3):
        window = tokens[i : i + 3]
        random.shuffle(window)
        tokens[i : i + 3] = window
    return tokens


# ------------------ datasets --------------------------------
class GenericPretrainDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        view1 = self.encode(augment(toks.copy()))
        view2 = self.encode(augment(toks.copy()))
        return view1, view2


class SPRDatasetCLS(Dataset):
    def __init__(self, seqs, labels_):
        self.seqs = seqs
        self.labels = [lab2id[l] for l in labels_]

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        t = self.encode(self.seqs[idx].split())
        return t, self.labels[idx], self.seqs[idx]


def pad_batch(list_of_seq):
    lens = [len(s) for s in list_of_seq]
    mx = max(lens)
    arr = np.full((len(list_of_seq), mx), pad_id, dtype=np.int64)
    for i, s in enumerate(list_of_seq):
        arr[i, : len(s)] = s
    return torch.tensor(arr), torch.tensor(lens)


def collate_pretrain(batch):
    v1, v2 = zip(*batch)
    a, lena = pad_batch(v1)
    b, lenb = pad_batch(v2)
    return (a, lena, b, lenb)


def collate_cls(batch):
    seqs, ys, raw = zip(*batch)
    x, lens = pad_batch(seqs)
    return (x, lens, torch.tensor(ys)), list(raw)


# ------------------ model -----------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, 128)

    def forward(self, x, lens, project=True):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.proj(h) if project else h


class Classifier(nn.Module):
    def __init__(self, encoder, nclass):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(128, nclass)

    def forward(self, x, lens):
        z = self.enc(x, lens, project=True)
        return self.head(z)


# ------------------ contrastive loss ------------------------
def nt_xent(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    N = z1.size(0)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss = nn.functional.cross_entropy(sim[:N], targets) + nn.functional.cross_entropy(
        sim[N:], targets - N
    )
    return loss * 0.5


# ------------------ pretraining -----------------------------
def pretrain_encoder(epochs=3, batch=256, lr=1e-3):
    enc = Encoder(len(vocab)).to(device)
    loader = DataLoader(
        GenericPretrainDataset(combined_train_sequences),
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_pretrain,
    )
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    enc.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss_sum = 0
        for a, lena, b, lenb in loader:
            a, lena, b, lenb = (
                a.to(device),
                lena.to(device),
                b.to(device),
                lenb.to(device),
            )
            z1 = enc(a, lena)
            z2 = enc(b, lenb)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * a.size(0)
        print(
            f"Pretrain epoch {ep}: loss={(loss_sum/len(loader.dataset)):.4f}  "
            f"time={time.time()-t0:.1f}s"
        )
    return enc


pretrained_enc = pretrain_encoder()

# ------------------ fine-tuning -----------------------------
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
    preds, gt, rawseq = [], [], []
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
    # logging
    experiment_data["MultiDatasetContrastivePretrain"]["SPR_BENCH"]["losses"][
        "train"
    ].append(tr_loss)
    experiment_data["MultiDatasetContrastivePretrain"]["SPR_BENCH"]["losses"][
        "val"
    ].append(val_loss)
    experiment_data["MultiDatasetContrastivePretrain"]["SPR_BENCH"]["metrics"][
        "val"
    ].append(cwa)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}  CompWA={cwa:.4f}")

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        wait = 0
        experiment_data["MultiDatasetContrastivePretrain"]["SPR_BENCH"][
            "predictions"
        ] = preds
        experiment_data["MultiDatasetContrastivePretrain"]["SPR_BENCH"][
            "ground_truth"
        ] = gt
        best_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        break

model.load_state_dict(best_state)

# ------------------ save ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training finished. Data saved to ./working/experiment_data.npy")
