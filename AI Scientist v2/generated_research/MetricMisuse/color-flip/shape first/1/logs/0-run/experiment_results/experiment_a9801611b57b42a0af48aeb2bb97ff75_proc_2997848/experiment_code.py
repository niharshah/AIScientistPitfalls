import os, pathlib, random, gc, math, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ---------- set up working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- small utilities ----------
def tokenize(seq):  # token -> e.g. "A3"
    return seq.strip().split()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# try to find real data, else synthesize tiny one
data_roots = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_roots:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded SPR_BENCH from {p}")
        break
if spr_bench is None:
    print("No SPR_BENCH found – synthesising toy data.")

    def synth(n):
        sh, co = "ABCDE", "12345"
        seqs, labs = [], []
        for i in range(n):
            L = random.randint(4, 9)
            seqs.append(
                " ".join(random.choice(sh) + random.choice(co) for _ in range(L))
            )
            labs.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labs}

    spr_bench = DatasetDict()
    for split, n in [("train", 800), ("dev", 200), ("test", 200)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )

# -------------- build separate vocabularies ---------------
shape_vocab = {"<PAD>": 0}
color_vocab = {"<PAD>": 0}
for seq in spr_bench["train"]["sequence"]:
    for tok in tokenize(seq):
        s, c = tok[0], tok[1]
        if s not in shape_vocab:
            shape_vocab[s] = len(shape_vocab)
        if c not in color_vocab:
            color_vocab[c] = len(color_vocab)
pad_id = 0
print(f"Shape vocab: {len(shape_vocab)}  Color vocab: {len(color_vocab)}")


# ------------------ metrics -----------------
def count_shape_variety(seq):
    return len(set(t[0] for t in tokenize(seq)))


def count_color_variety(seq):
    return len(set(t[1] for t in tokenize(seq)))


def shape_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w) if w else 0


def color_weighted_accuracy(seqs, y, g):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w) if w else 0


def complexity_weighted_accuracy(seqs, y, g):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w) if w else 0


# -------------- datasets & augmentation -------------------
class ContrastiveSPR(Dataset):
    def __init__(self, hfds):
        self.seqs = hfds["sequence"]

    def _augment(self, toks):
        # simple random deletion + span mask
        if len(toks) == 1:
            return toks
        out = []
        i = 0
        while i < len(toks):
            if random.random() < 0.1:  # delete token
                i += 1
                continue
            if random.random() < 0.05:  # mask span of length k
                k = min(len(toks) - i, random.randint(1, 3))
                out.extend(["<PAD>"] * k)
                i += k
            else:
                out.append(toks[i])
                i += 1
        if not out:
            out = [random.choice(toks)]
        return out

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tok = tokenize(self.seqs[idx])
        return self._augment(tok), self._augment(tok)


class ClassifierSPR(Dataset):
    def __init__(self, hfds):
        self.seqs = hfds["sequence"]
        self.labels = hfds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def encode_tokens(tok_list):
    shape_ids = [shape_vocab.get(t[0], 0) for t in tok_list]
    color_ids = [color_vocab.get(t[1], 0) for t in tok_list]
    return torch.tensor(shape_ids), torch.tensor(color_ids)


def collate_contrastive(batch):
    a1, a2 = zip(*batch)

    def encode(list_of_tok_lists):
        sids = [encode_tokens(t)[0] for t in list_of_tok_lists]
        cids = [encode_tokens(t)[1] for t in list_of_tok_lists]
        lens = [len(x) for x in sids]
        sids = pad_sequence(sids, batch_first=True, padding_value=0)
        cids = pad_sequence(cids, batch_first=True, padding_value=0)
        return sids.to(device), cids.to(device), torch.tensor(lens).to(device)

    s1, c1, l1 = encode(a1)
    s2, c2, l2 = encode(a2)
    return {"s1": s1, "c1": c1, "l1": l1, "s2": s2, "c2": c2, "l2": l2}


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    sids = [encode_tokens(t)[0] for t in toks]
    cids = [encode_tokens(t)[1] for t in toks]
    lens = [len(x) for x in sids]
    sids = pad_sequence(sids, batch_first=True, padding_value=0).to(device)
    cids = pad_sequence(cids, batch_first=True, padding_value=0).to(device)
    return {
        "s": sids,
        "c": cids,
        "l": torch.tensor(lens).to(device),
        "y": torch.tensor(labels).to(device),
        "seq": raw,
    }


# ------------------ model -------------------
class CompositionalEncoder(nn.Module):
    def __init__(
        self, shape_vocab_size, color_vocab_size, d_model=64, nhead=4, layers=2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(shape_vocab_size, d_model, padding_idx=pad_id)
        self.color_emb = nn.Embedding(color_vocab_size, d_model, padding_idx=pad_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, shape_ids, color_ids, lens):
        x = self.shape_emb(shape_ids) + self.color_emb(color_ids)
        # create padding mask
        mask = shape_ids == pad_id
        h = self.transformer(x, src_key_padding_mask=mask)
        # masked mean pool
        mask_f = (~mask).unsqueeze(-1)
        h = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return h


class SPRClassifier(nn.Module):
    def __init__(self, encoder, classes=2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.shape_emb.embedding_dim, classes)

    def forward(self, shape_ids, color_ids, lens):
        z = self.encoder(shape_ids, color_ids, lens)
        return self.head(z)


def nt_xent(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2N x d
    sim = torch.exp(torch.mm(z, z.t()) / temp)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim = sim.masked_fill(mask, 0)
    pos = torch.cat([torch.diag(sim[:N, N:]), torch.diag(sim[N:, :N])])
    denom = sim.sum(1)
    loss = -torch.log(pos / denom).mean()
    return loss


# -------------- experiment data container ------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "SWA": [],
            "CWA": [],
            "CompWA": [],
        },
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------- data loaders -------------------
pretrain_loader = DataLoader(
    ContrastiveSPR(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    ClassifierSPR(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    ClassifierSPR(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)

# ------------------ build model ----------------------
encoder = CompositionalEncoder(len(shape_vocab), len(color_vocab)).to(device)
clf = SPRClassifier(encoder, classes=len(set(spr_bench["train"]["label"]))).to(device)

# ------------------ optimizers -----------------------
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------- contrastive pretrain --------------
pretrain_epochs = 4
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    epoch_loss = 0
    for batch in pretrain_loader:
        opt_pt.zero_grad()
        z1 = encoder(batch["s1"], batch["c1"], batch["l1"])
        z2 = encoder(batch["s2"], batch["c2"], batch["l2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        opt_pt.step()
        epoch_loss += loss.item() * batch["s1"].size(0)
    print(
        f"Pretrain epoch {ep}/{pretrain_epochs}: loss={epoch_loss/len(pretrain_loader.dataset):.4f}"
    )

# ---------------- fine-tuning -----------------------
ft_epochs = 3
for ep in range(1, ft_epochs + 1):
    # ---- train ----
    clf.train()
    run_loss = 0
    for batch in train_loader:
        opt_ft.zero_grad()
        logits = clf(batch["s"], batch["c"], batch["l"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        opt_ft.step()
        run_loss += loss.item() * batch["s"].size(0)
    train_loss = run_loss / len(train_loader.dataset)
    # ---- validate ----
    clf.eval()
    val_loss, preds, gts, seqs = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = clf(batch["s"], batch["c"], batch["l"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["s"].size(0)
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p)
            gts.extend(batch["y"].cpu().tolist())
            seqs.extend(batch["seq"])
    val_loss /= len(dev_loader.dataset)
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    comp = complexity_weighted_accuracy(seqs, gts, preds)
    # ---- log ----
    experiment_data["SPR_BENCH"]["metrics"]["train_loss"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["CompWA"].append(comp)
    print(
        f"Epoch {ep}/{ft_epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

# store final preds
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# -------------- save experiment data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to ./working/experiment_data.npy")

# optional quick plot of val loss progression
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["metrics"]["val_loss"], label="val_loss")
plt.xlabel("Fine-tune epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "val_loss_curve.png"))
plt.close()

# tidy up
torch.cuda.empty_cache()
gc.collect()
