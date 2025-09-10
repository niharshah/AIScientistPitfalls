import os, pathlib, random, time, gc, math, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ----------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- load SPR_BENCH --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


possible_roots = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in possible_roots:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded SPR_BENCH from {p}")
        break
# synthetic fallback (small) -------------------------------------------------
if spr_bench is None:
    print("Dataset not found – creating tiny synthetic placeholder")
    shapes, colors = "ABCD", "1234"

    def synth(n):
        seqs, labels = [], []
        for i in range(n):
            L = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 800), ("dev", 200), ("test", 200)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )

# --------------- vocabulary -----------------
vocab = {"<PAD>": 0, "<MASK>": 1}
for seq in spr_bench["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id, mask_id = 0, 1
print("Vocab size:", len(vocab))


# --------------- metrics --------------------
def count_shape_variety(s):
    return len(set(t[0] for t in s.split()))


def count_color_variety(s):
    return len(set(t[1] for t in s.split()))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if w else 0


# ------------- augmentation utils ----------
def augment(tokens):
    out = []
    for t in tokens:
        r = random.random()
        if r < 0.15:  # 15% mask
            out.append("<MASK>")
        elif r < 0.25:  # 10% drop token
            continue
        else:
            out.append(t)
    # local permutation (context aware)
    window = 3
    i = 0
    while i < len(out):
        if random.random() < 0.3:
            j = min(len(out), i + window)
            segment = out[i:j]
            random.shuffle(segment)
            out[i:j] = segment
        i += window
    return out if out else ["<MASK>"]


# -------------- datasets --------------------
def tokens2ids(tok_list):
    return torch.tensor([vocab.get(t, mask_id) for t in tok_list], dtype=torch.long)


class ContrastiveSPR(Dataset):
    def __init__(self, hfds):
        self.seqs = hfds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        return augment(toks), augment(toks)


class ClassifierSPR(Dataset):
    def __init__(self, hfds):
        self.seqs = hfds["sequence"]
        self.labels = hfds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx].split(), self.labels[idx], self.seqs[idx]


def collate_con(batch):
    v1, v2 = zip(*batch)
    ids1 = [tokens2ids(t) for t in v1]
    ids2 = [tokens2ids(t) for t in v2]
    len1 = [len(i) for i in ids1]
    len2 = [len(i) for i in ids2]
    ids1 = pad_sequence(ids1, batch_first=True, padding_value=pad_id).to(device)
    ids2 = pad_sequence(ids2, batch_first=True, padding_value=pad_id).to(device)
    return {
        "ids1": ids1,
        "ids2": ids2,
        "len1": torch.tensor(len1).to(device),
        "len2": torch.tensor(len2).to(device),
    }


def collate_clf(batch):
    toks, labels, raw = zip(*batch)
    ids = [tokens2ids(t) for t in toks]
    lens = [len(i) for i in ids]
    ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device)
    return {
        "ids": ids,
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(labels).to(device),
        "seq": raw,
    }


# -------------- model -----------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_sz, d_model=96, nhead=4, num_layers=2, dim_feed=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model))  # maximum length 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feed, 0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, ids, lens=None):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        mask = ids == pad_id
        h = self.enc(x, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(
            min=1
        )  # mean pooling
        return self.proj(h)


class SPRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# -------------- loss ------------------------
def nt_xent(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = (z @ z.T) / temp
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(N, 2 * N, device=z.device)
    logits12 = sim[:N, N:]
    logits21 = sim[N:, :N]
    loss = (
        -(logits12.diag().mean() + logits21.diag().mean()) / 2
        + (torch.logsumexp(sim[:N], 1).mean() + torch.logsumexp(sim[N:], 1).mean()) / 2
    )
    return loss


# ------------ experiment store --------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- dataloaders ---------------
pre_loader = DataLoader(
    ContrastiveSPR(spr_bench["train"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_con,
)
train_loader = DataLoader(
    ClassifierSPR(spr_bench["train"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_clf,
)
dev_loader = DataLoader(
    ClassifierSPR(spr_bench["dev"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_clf,
)

# -------------- build model -----------------
encoder = TransformerEncoder(len(vocab)).to(device)
clf = SPRClassifier(encoder, num_classes=len(set(spr_bench["train"]["label"]))).to(
    device
)

# -------------- optimizers ------------------
opt_pre = torch.optim.Adam(encoder.parameters(), lr=1e-3)
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()

# -------------- pretrain --------------------
pre_epochs = 5
for epoch in range(1, pre_epochs + 1):
    encoder.train()
    running = 0
    for b in pre_loader:
        opt_pre.zero_grad()
        z1 = encoder(b["ids1"], b["len1"])
        z2 = encoder(b["ids2"], b["len2"])
        loss = nt_xent(z1, z2)
        loss.backward()
        opt_pre.step()
        running += loss.item() * b["ids1"].size(0)
    ep_loss = running / len(pre_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(ep_loss)
    print(f"Pretrain epoch {epoch}/{pre_epochs} loss={ep_loss:.4f}")

# -------------- fine-tune --------------------
ft_epochs = 4
for epoch in range(1, ft_epochs + 1):
    # train
    clf.train()
    tr_loss = 0
    for b in train_loader:
        opt_ft.zero_grad()
        logit = clf(b["ids"], b["len"])
        loss = ce_loss(logit, b["label"])
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item() * b["ids"].size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # validate
    clf.eval()
    val_loss = 0
    all_p = []
    all_t = []
    all_seq = []
    with torch.no_grad():
        for b in dev_loader:
            logit = clf(b["ids"], b["len"])
            loss = ce_loss(logit, b["label"])
            val_loss += loss.item() * b["ids"].size(0)
            p = logit.argmax(1).cpu().tolist()
            all_p.extend(p)
            all_t.extend(b["label"].cpu().tolist())
            all_seq.extend(b["seq"])
    val_loss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    swa = shape_weighted_accuracy(all_seq, all_t, all_p)
    cwa = color_weighted_accuracy(all_seq, all_t, all_p)
    comp = complexity_weighted_accuracy(all_seq, all_t, all_p)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": epoch, "swa": swa, "cwa": cwa, "comp": comp}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "val_loss": val_loss, "swa": swa, "cwa": cwa, "comp": comp}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

# store final predictions/gt for analysis
experiment_data["SPR_BENCH"]["predictions"] = all_p
experiment_data["SPR_BENCH"]["ground_truth"] = all_t

# -------------- save everything -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
