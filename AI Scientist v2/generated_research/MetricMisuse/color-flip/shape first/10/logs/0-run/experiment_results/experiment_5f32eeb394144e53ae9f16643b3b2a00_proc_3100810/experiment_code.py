import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────────── housekeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "losses": {"contrastive": [], "train": [], "val": []},
        "metrics": {
            "SWA_train": [],
            "CWA_train": [],
            "SCWA_train": [],
            "SWA_val": [],
            "CWA_val": [],
            "SCWA_val": [],
        },
        "predictions": [],
        "ground_truth": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ───────────────────────────────── metrics
def count_shape_variety(seq):  # e.g. token "A1" -> shape A
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def SWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def CWA(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


def SCWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# ───────────────────────────────── data loading
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic data.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:6])
        colors = list("123456")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seqs.append(" ".join(toks))
            labels.append(
                max(
                    set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
                )
            )
        ids = list(range(n))
        return {"id": ids, "sequence": seqs, "label": labels}

    import datasets

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(4000)),
            "dev": datasets.Dataset.from_dict(synth_split(800)),
            "test": datasets.Dataset.from_dict(synth_split(800)),
        }
    )

# ───────────────────────────────── vocab
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(ds):
    global token2id, label2id
    tokens, labels = set(), set()
    for s, l in zip(ds["sequence"], ds["label"]):
        tokens.update(s.split())
        labels.add(l)
    token2id = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}
    label2id = {lab: i for i, lab in enumerate(sorted(labels))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_seq(s):
    return [token2id[t] for t in s.split()]


def encode_lab(l):
    return label2id[l]


# ───────────────────────────────── datasets
def mask_or_shuffle(tokens, p_mask=0.2, p_shuffle=0.3):
    toks = tokens.copy()
    # mask
    for i in range(len(toks)):
        if random.random() < p_mask:
            toks[i] = PAD_ID
    # local shuffle
    if random.random() < p_shuffle and len(toks) > 3:
        i = random.randint(0, len(toks) - 3)
        window = toks[i : i + 3]
        random.shuffle(window)
        toks[i : i + 3] = window
    return toks


class ContrastiveDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = encode_seq(self.seqs[idx])
        aug1 = mask_or_shuffle(toks)
        aug2 = mask_or_shuffle(toks)
        return torch.tensor(aug1, dtype=torch.long), torch.tensor(
            aug2, dtype=torch.long
        )


def contrastive_collate(batch):
    view1, view2 = zip(*batch)
    all_views = list(view1) + list(view2)
    max_len = max(len(v) for v in all_views)
    padded = torch.zeros(len(all_views), max_len, dtype=torch.long)
    for i, v in enumerate(all_views):
        padded[i, : len(v)] = v
    return padded  # size 2B x L


class ClassifyDataset(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(encode_seq(self.seq[idx]), dtype=torch.long),
            "lab": torch.tensor(encode_lab(self.lab[idx]), dtype=torch.long),
            "raw": self.seq[idx],
        }


def class_collate(batch):
    max_len = max(len(x["ids"]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labs = torch.zeros(len(batch), dtype=torch.long)
    raws = []
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
        labs[i] = b["lab"]
        raws.append(b["raw"])
    return {"ids": ids, "lab": labs, "raw": raws}


# ───────────────────────────────── model
class Encoder(nn.Module):
    def __init__(self, vocab, dim=128, nhead=4, nlayers=2, max_len=50):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(max_len, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, batch_first=True
        )
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        B, L = x.size()
        if L > self.pos.size(0):  # extend positions if necessary
            extra = L - self.pos.size(0)
            self.pos.data = torch.cat(
                [self.pos.data, torch.randn(extra, self.dim, device=self.pos.device)]
            )
        out = self.embed(x) + self.pos[:L]
        mask = x == PAD_ID
        h = self.trans(out, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1
        ).unsqueeze(-1)
        return h  # B x dim


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(encoder.dim, num_labels)

    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            h = self.enc(x)
        return self.fc(h)


# ───────────────────────────────── contrastive helpers
def nt_xent(z, batch, temp=0.2):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp  # 2B x 2B
    B = batch
    labels = torch.arange(0, 2 * B, device=z.device)
    pos = labels + torch.where(
        labels % 2 == 0,
        torch.tensor(1, device=z.device),
        torch.tensor(-1, device=z.device),
    )
    numerator = torch.exp(sim[labels, pos])
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    denom = torch.exp(sim)[mask].view(2 * B, -1).sum(1)
    loss = -torch.log(numerator / denom).mean()
    return loss


# ───────────────────────────────── loaders
CONTRAST_BATCH = 256
contrast_loader = DataLoader(
    ContrastiveDataset(spr["train"]),
    batch_size=CONTRAST_BATCH // 2,
    shuffle=True,
    collate_fn=contrastive_collate,
)

train_loader = DataLoader(
    ClassifyDataset(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=class_collate,
)
dev_loader = DataLoader(
    ClassifyDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=class_collate
)

# ───────────────────────────────── stage 1: contrastive pretraining
enc = Encoder(len(token2id) + 1, dim=128).to(device)
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
EPOCHS_CON = 3
for ep in range(1, EPOCHS_CON + 1):
    enc.train()
    tot = 0
    n = 0
    for views in contrast_loader:
        views = views.to(device)
        B = views.size(0) // 2
        opt.zero_grad()
        z = enc(views)  # (2B) x dim
        loss = nt_xent(z, B)
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    l = tot / n
    experiment_data["SPR_BENCH"]["losses"]["contrastive"].append(l)
    print(f"Contrastive epoch {ep}: loss={l:.4f}")

# ───────────────────────────────── stage 2: fine-tune classifier
clf = Classifier(enc, len(label2id)).to(device)
criterion = nn.CrossEntropyLoss()
opt_cls = torch.optim.Adam(clf.parameters(), lr=2e-4)
EPOCHS_CLS = 5
for ep in range(1, EPOCHS_CLS + 1):
    # train
    clf.train()
    t_loss = 0
    for batch in train_loader:
        ids = batch["ids"].to(device)
        labs = batch["lab"].to(device)
        opt_cls.zero_grad()
        logits = clf(ids)
        loss = criterion(logits, labs)
        loss.backward()
        opt_cls.step()
        t_loss += loss.item() * ids.size(0)
    t_loss /= len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(t_loss)
    # evaluate
    clf.eval()
    v_loss = 0
    y_true = y_pred = seqs = []
    y_true = []
    y_pred = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["ids"].to(device)
            labs = batch["lab"].to(device)
            logits = clf(ids)
            loss = criterion(logits, labs)
            v_loss += loss.item() * ids.size(0)
            preds = logits.argmax(1).cpu().tolist()
            labs = labs.cpu().tolist()
            y_true.extend([id2label[i] for i in labs])
            y_pred.extend([id2label[i] for i in preds])
            seqs.extend(batch["raw"])
    v_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    swa = CWA(seqs, y_true, y_pred)  # note: want all
    cwa = CWA(seqs, y_true, y_pred)
    scwa = SCWA(seqs, y_true, y_pred)
    experiment_data["SPR_BENCH"]["metrics"]["SWA_val"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA_val"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA_val"].append(scwa)
    print(
        f"Epoch {ep}: validation_loss = {v_loss:.4f} | SWA {swa:.3f} CWA {cwa:.3f} SCWA {scwa:.3f}"
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
