import os, pathlib, random, math, gc, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------  SPR loader (from baseline) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded SPR_BENCH from {p}")
        break
assert spr_bench is not None, "SPR_BENCH dataset not found!"


# --------------------  tokenisation / vocab ---------------
def tok(seq):
    return seq.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2}
for split in spr_bench.values():
    for s in split["sequence"]:
        for t in tok(s):
            if t not in vocab:
                vocab[t] = len(vocab)
pad_id, mask_id, cls_id = vocab["<PAD>"], vocab["<MASK>"], vocab["<CLS>"]
vocab_size = len(vocab)
print("Vocab size =", vocab_size)

max_len = (
    max(len(tok(s)) for split in spr_bench.values() for s in split["sequence"]) + 1
)  # +CLS


# -------------------- metrics -----------------------------
def count_shape_variety(seq):
    return len(set(t[0] for t in tok(seq)))


def count_color_variety(seq):
    return len(set(t[1] for t in tok(seq)))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------- datasets ----------------------------
class SPRContrastive(Dataset):
    def __init__(self, ds):
        self.seqs = ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, tokens):
        out = []
        for t in tokens:
            r = random.random()
            if r < 0.1:  # deletion
                continue
            elif r < 0.2:  # masking
                out.append("<MASK>")
            else:
                out.append(t)
        # local shuffle with prob 0.3
        if random.random() < 0.3 and len(out) > 2:
            i = random.randrange(len(out) - 1)
            out[i], out[i + 1] = out[i + 1], out[i]
        return out if out else ["<MASK>"]

    def __getitem__(self, idx):
        toks = tok(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class SPRClassify(Dataset):
    def __init__(self, ds):
        self.seqs, self.labels = ds["sequence"], ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tok(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def encode_batch(token_lists):
    ids = [
        torch.tensor([cls_id] + [vocab.get(t, mask_id) for t in tl], dtype=torch.long)
        for tl in token_lists
    ]
    lens = [len(i) for i in ids]
    return pad_sequence(ids, batch_first=True, padding_value=pad_id), torch.tensor(lens)


def collate_contrastive(batch):
    t1, t2 = zip(*batch)
    ids1, l1 = encode_batch(t1)
    ids2, l2 = encode_batch(t2)
    return {
        "ids1": ids1.to(device),
        "len1": l1.to(device),
        "ids2": ids2.to(device),
        "len2": l2.to(device),
    }


def collate_classify(batch):
    toks, lab, raw = zip(*batch)
    ids, lens = encode_batch(toks)
    return {
        "ids": ids.to(device),
        "len": lens.to(device),
        "label": torch.tensor(lab, dtype=torch.long).to(device),
        "sequence": raw,
    }


# ------------------  Transformer encoder ------------------
class SymbolicTransformer(nn.Module):
    def __init__(self, vocab_sz, d_model=96, n_head=4, n_layer=2, dim_ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.randn(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=dim_ff, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, n_layer)
        self.out_dim = d_model

    def forward(self, ids, lens):
        x = self.emb(ids) + self.pos[: ids.size(1)]
        mask = ids == pad_id
        h = self.tr(x, src_key_padding_mask=mask)
        return h[:, 0]  # CLS token


class Classifier(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, n_cls)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# ------------------  Contrastive loss ---------------------
def nt_xent(z1, z2, T=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.mm(z, z.t()) / T
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, device=z.device)
    loss = (
        nn.functional.cross_entropy(sim[:N, N:], targets)
        + nn.functional.cross_entropy(sim[N:, :N], targets)
    ) * 0.5
    return loss


# ------------------  experiment store ---------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------  loaders ------------------------------
bt_con, bt_train, bt_eval = 128, 128, 256
pretrain_loader = DataLoader(
    SPRContrastive(spr_bench["train"]),
    batch_size=bt_con,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassify(spr_bench["train"]),
    batch_size=bt_train,
    shuffle=True,
    collate_fn=collate_classify,
)
dev_loader = DataLoader(
    SPRClassify(spr_bench["dev"]),
    batch_size=bt_eval,
    shuffle=False,
    collate_fn=collate_classify,
)

# ------------------  model --------------------------------
encoder = SymbolicTransformer(vocab_size).to(device)
clf = Classifier(encoder, n_cls=len(set(spr_bench["train"]["label"]))).to(device)

# ------------------  contrastive pre-training -------------
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pretrain_epochs = 6
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    total = 0
    s = 0.0
    for b in pretrain_loader:
        opt_pt.zero_grad()
        z1 = encoder(b["ids1"], b["len1"])
        z2 = encoder(b["ids2"], b["len2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        opt_pt.step()
        s += loss.item() * b["ids1"].size(0)
        total += b["ids1"].size(0)
    epoch_loss = s / total
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(epoch_loss)
    print(f"Pretrain epoch {ep}/{pretrain_epochs} loss={epoch_loss:.4f}")

# ------------------  fine-tuning --------------------------
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
ft_epochs = 5
for ep in range(1, ft_epochs + 1):
    # train
    clf.train()
    s = 0.0
    total = 0
    for b in train_loader:
        opt_ft.zero_grad()
        logits = clf(b["ids"], b["len"])
        loss = ce(logits, b["label"])
        loss.backward()
        opt_ft.step()
        s += loss.item() * b["ids"].size(0)
        total += b["ids"].size(0)
    tr_loss = s / total
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # validation
    clf.eval()
    s = 0.0
    total = 0
    all_pred = []
    all_true = []
    all_seq = []
    with torch.no_grad():
        for b in dev_loader:
            logits = clf(b["ids"], b["len"])
            loss = ce(logits, b["label"])
            s += loss.item() * b["ids"].size(0)
            total += b["ids"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(b["label"].cpu().tolist())
            all_seq.extend(b["sequence"])
    val_loss = s / total
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    comp = complexity_weighted_accuracy(all_seq, all_true, all_pred)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": ep, "train_loss": tr_loss}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": ep, "val_loss": val_loss, "SWA": swa, "CWA": cwa, "CompWA": comp}
    )
    experiment_data["SPR_BENCH"]["predictions"] = all_pred
    experiment_data["SPR_BENCH"]["ground_truth"] = all_true
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

# ------------------  save artefacts -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)

# save simple loss curve
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.xlabel("Fine-tune epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "ft_loss_curve.png"))
plt.close()
