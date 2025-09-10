import os, pathlib, random, gc, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _ld(f"{sp}.csv")
    return out


data_roots = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr = None
for p in data_roots:
    if (p / "train.csv").exists():
        spr = load_spr_bench(p)
        print(f"Loaded real SPR_BENCH from {p}")
        break
if spr is None:  # fallback tiny synthetic
    print("Real SPR_BENCH not found – creating tiny synthetic dataset")

    def synth(n):
        sh, col = "ABCD", "1234"
        seq, lab = [], []
        for i in range(n):
            L = random.randint(4, 8)
            seq.append(
                " ".join(random.choice(sh) + random.choice(col) for _ in range(L))
            )
            lab.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seq, "label": lab}

    spr = DatasetDict()
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        spr[sp] = load_dataset("json", data_files=None, split="train", data=synth(n))

# ---------- vocabularies ----------
tok2id = {"<PAD>": 0, "<MASK>": 1}
shape2id = {"<PAD>": 0}
color2id = {"<PAD>": 0}


def register_token(tok: str):
    if tok not in tok2id:
        tok2id[tok] = len(tok2id)
    sh, col = tok[0], (tok[1] if len(tok) > 1 else "0")
    if sh not in shape2id:
        shape2id[sh] = len(shape2id)
    if col not in color2id:
        color2id[col] = len(color2id)


for split in spr.values():
    for seq in split["sequence"]:
        for t in seq.strip().split():
            register_token(t)
# also register mask pseudo token
shape2id["?"] = len(shape2id)
color2id["?"] = len(color2id)

print(f"Token vocab: {len(tok2id)} | shapes: {len(shape2id)} | colors: {len(color2id)}")


# ---------- metrics ----------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def count_color_variety(seq):
    return len(set((t[1] if len(t) > 1 else "0") for t in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(1, sum(w))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(1, sum(w))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(1, sum(w))


# ---------- augmentation ----------
def augment_tokens(toks):
    toks = toks[:]  # copy
    # 15% masking
    out = []
    for t in toks:
        if random.random() < 0.15:
            out.append("<MASK>")
        else:
            out.append(t)
    # random single swap
    if len(out) > 1 and random.random() < 0.3:
        i = random.randint(0, len(out) - 2)
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


# ---------- datasets ----------
class ContrastiveDS(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        return augment_tokens(toks), augment_tokens(toks)


class ClassifyDS(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx].split(), self.labels[idx], self.seqs[idx]


# ---------- collators ----------
def encode_lists(token_lists):
    tok_ids, sh_ids, col_ids, lens = [], [], [], []
    for toks in token_lists:
        ids, sh, co = [], [], []
        for t in toks:
            tid = tok2id.get(t, tok2id["<MASK>"])
            sid = shape2id.get((t[0] if t not in ["<MASK>"] else "?"), shape2id["?"])
            cid = color2id.get(
                (t[1] if len(t) > 1 and t not in ["<MASK>"] else "?"), color2id["?"]
            )
            ids.append(tid)
            sh.append(sid)
            co.append(cid)
        tok_ids.append(torch.tensor(ids))
        sh_ids.append(torch.tensor(sh))
        col_ids.append(torch.tensor(co))
        lens.append(len(ids))
    pad = lambda seqs: pad_sequence(seqs, batch_first=True, padding_value=0)
    return pad(tok_ids), pad(sh_ids), pad(col_ids), torch.tensor(lens)


def collate_contrastive(batch):
    v1, v2 = zip(*batch)
    ids1, sh1, col1, len1 = encode_lists(v1)
    ids2, sh2, col2, len2 = encode_lists(v2)
    batch_tensors = {
        "ids1": ids1.to(device),
        "sh1": sh1.to(device),
        "col1": col1.to(device),
        "len1": len1.to(device),
        "ids2": ids2.to(device),
        "sh2": sh2.to(device),
        "col2": col2.to(device),
        "len2": len2.to(device),
    }
    return batch_tensors


def collate_classifier(batch):
    toks, lbls, raw_seq = zip(*batch)
    ids, sh, col, lens = encode_lists(toks)
    return {
        "ids": ids.to(device),
        "sh": sh.to(device),
        "col": col.to(device),
        "len": lens.to(device),
        "label": torch.tensor(lbls, dtype=torch.long).to(device),
        "sequence": raw_seq,
    }


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, d_tok=32, d_model=64):
        super().__init__()
        self.token_emb = nn.Embedding(len(tok2id), d_tok, padding_idx=0)
        self.shape_emb = nn.Embedding(len(shape2id), d_tok, padding_idx=0)
        self.color_emb = nn.Embedding(len(color2id), d_tok, padding_idx=0)
        self.proj = nn.Linear(d_tok * 3, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.out = nn.Linear(d_model * 2, d_model)

    def forward(self, ids, sh, col, lens):
        x = torch.cat(
            [self.token_emb(ids), self.shape_emb(sh), self.color_emb(col)], dim=-1
        )
        x = self.proj(x)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.out(h)  # (B, d_model)


class Classifier(nn.Module):
    def __init__(self, enc, n_classes):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.out.out_features, n_classes)

    def forward(self, ids, sh, col, lens):
        z = self.enc(ids, sh, col, lens)
        return self.head(z)


# ---------- contrastive loss ----------
def info_nce(z1, z2, temp=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = (z @ z.T) / temp
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss = (
        nn.functional.cross_entropy(sim[:N], targets)
        + nn.functional.cross_entropy(sim[N:], targets - N)
    ) * 0.5
    return loss


# ---------- dataloaders ----------
batch_c = 256
pt_loader = DataLoader(
    ContrastiveDS(spr["train"]),
    batch_size=batch_c,
    shuffle=True,
    collate_fn=collate_contrastive,
)
tr_loader = DataLoader(
    ClassifyDS(spr["train"]),
    batch_size=batch_c,
    shuffle=True,
    collate_fn=collate_classifier,
)
dv_loader = DataLoader(
    ClassifyDS(spr["dev"]),
    batch_size=batch_c,
    shuffle=False,
    collate_fn=collate_classifier,
)

# ---------- experiment dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------- model / optim ----------
encoder = Encoder().to(device)
clf = Classifier(encoder, n_classes=len(set(spr["train"]["label"]))).to(device)

# ---------- pre-training ----------
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pretrain_epochs = 4
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    running = 0.0
    for batch in pt_loader:
        opt_pt.zero_grad()
        z1 = encoder(batch["ids1"], batch["sh1"], batch["col1"], batch["len1"])
        z2 = encoder(batch["ids2"], batch["sh2"], batch["col2"], batch["len2"])
        loss = info_nce(z1, z2)
        loss.backward()
        opt_pt.step()
        running += loss.item() * batch["ids1"].size(0)
    epoch_loss = running / len(pt_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain epoch {ep}/{pretrain_epochs} loss={epoch_loss:.4f}")

# ---------- fine-tuning ----------
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
ft_epochs = 3
for ep in range(1, ft_epochs + 1):
    # train
    clf.train()
    run_tr = 0.0
    for batch in tr_loader:
        opt_ft.zero_grad()
        logits = clf(batch["ids"], batch["sh"], batch["col"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        opt_ft.step()
        run_tr += loss.item() * batch["ids"].size(0)
    tr_loss = run_tr / len(tr_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # val
    clf.eval()
    run_val = 0.0
    all_p, all_t, all_s = [], [], []
    with torch.no_grad():
        for batch in dv_loader:
            logits = clf(batch["ids"], batch["sh"], batch["col"], batch["len"])
            loss = criterion(logits, batch["label"])
            run_val += loss.item() * batch["ids"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_p.extend(preds)
            all_t.extend(batch["label"].cpu().tolist())
            all_s.extend(batch["sequence"])
    val_loss = run_val / len(dv_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    swa = shape_weighted_accuracy(all_s, all_t, all_p)
    cwa = color_weighted_accuracy(all_s, all_t, all_p)
    comp = complexity_weighted_accuracy(all_s, all_t, all_p)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((swa, cwa, comp))
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
