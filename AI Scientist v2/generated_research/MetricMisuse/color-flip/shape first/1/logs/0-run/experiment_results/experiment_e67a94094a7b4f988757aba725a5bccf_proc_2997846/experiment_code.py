import os, pathlib, random, math, gc, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# --------------------  mandatory working dir -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------  device ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------  load SPR_BENCH -------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


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
if spr_bench is None:  # fallback tiny synthetic
    print("SPR_BENCH not found – creating toy data")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seq, lbl = [], []
        for i in range(n):
            L = random.randint(4, 9)
            seq.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            lbl.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seq, "label": lbl}

    spr_bench = DatasetDict()
    for sp, n in [("train", 600), ("dev", 120), ("test", 120)]:
        spr_bench[sp] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )

# --------------------- vocab ----------------------------------------
tok2id = {"<PAD>": 0, "<MASK>": 1}


def tokenize(s):
    return s.strip().split()


for seq in spr_bench["train"]["sequence"]:
    for tok in tokenize(seq):
        if tok not in tok2id:
            tok2id[tok] = len(tok2id)
vocab_size = len(tok2id)
mask_id = tok2id["<MASK>"]
print("Vocab size:", vocab_size)


# -------------------- metrics ---------------------------------------
def count_shape(seq):
    return len({t[0] for t in tokenize(seq)})


def count_color(seq):
    return len({t[1] for t in tokenize(seq) if len(t) > 1})


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def complexity_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape(s) + count_color(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------- datasets --------------------------------------
class ContrastiveSPR(Dataset):
    def __init__(self, hfd):
        self.seqs = hfd["sequence"]

    def _augment(self, toks):
        # span mask
        toks = toks.copy()
        if len(toks) > 2:
            st = random.randint(0, len(toks) - 2)
            ln = random.randint(1, max(1, len(toks) // 3))
            for i in range(st, min(len(toks), st + ln)):
                toks[i] = "<MASK>"
        # local shuffle (k=3)
        k = 3
        if len(toks) > k:
            i = random.randint(0, len(toks) - k)
            window = toks[i : i + k]
            random.shuffle(window)
            toks[i : i + k] = window
        return toks

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class ClassifierSPR(Dataset):
    def __init__(self, hfd):
        self.seqs = hfd["sequence"]
        self.labels = hfd["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def pad_encode(token_lists):
    id_seqs = [
        torch.tensor([tok2id.get(t, mask_id) for t in tl], dtype=torch.long)
        for tl in token_lists
    ]
    lens = [len(t) for t in id_seqs]
    ids = pad_sequence(id_seqs, batch_first=True, padding_value=0)
    return ids, lens


def collate_contrastive(batch):
    a, b = zip(*batch)
    ids1, l1 = pad_encode(a)
    ids2, l2 = pad_encode(b)
    return {
        "ids1": ids1.to(device),
        "len1": torch.tensor(l1).to(device),
        "ids2": ids2.to(device),
        "len2": torch.tensor(l2).to(device),
    }


def collate_classifier(batch):
    toks, lab, raw = zip(*batch)
    ids, lens = pad_encode(toks)
    return {
        "ids": ids.to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(lab).to(device),
        "seq": raw,
    }


# -------------------- model ----------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_sz, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(512, d_model))  # enough for longest seq
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Linear(d_model, d_model)  # simple projection

    def forward(self, ids, lens):
        x = self.emb(ids) + self.pos[: ids.size(1)]
        mask = ids == 0
        h = self.transformer(x, src_key_padding_mask=mask)
        # mean pooling over valid tokens
        lens_tensor = torch.tensor(lens, device=ids.device).unsqueeze(1)
        h_sum = torch.sum(h.masked_fill(mask.unsqueeze(2), 0), dim=1)
        h_mean = h_sum / lens_tensor
        return self.pool(h_mean)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.pool.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# -------------------- loss ----------------------------------------
def info_nce(z1, z2, temp=0.2):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss1 = nn.functional.cross_entropy(sim[:N], targets)
    loss2 = nn.functional.cross_entropy(sim[N:], targets - N)
    return (loss1 + loss2) / 2


# -------------------- experiment container ------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------- dataloaders ---------------------------------
batch_pt = 256
batch_ft = 128
pre_loader = DataLoader(
    ContrastiveSPR(spr_bench["train"]),
    batch_size=batch_pt,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    ClassifierSPR(spr_bench["train"]),
    batch_size=batch_ft,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    ClassifierSPR(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)

# -------------------- build model ---------------------------------
encoder = TransformerEncoder(vocab_size).to(device)
clf = Classifier(encoder, num_classes=len(set(spr_bench["train"]["label"]))).to(device)

# -------------------- optimizers ----------------------------------
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()

# -------------------- pre-training --------------------------------
pretrain_epochs = 6
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    running = 0
    for batch in pre_loader:
        opt_pt.zero_grad()
        z1 = encoder(batch["ids1"], batch["len1"])
        z2 = encoder(batch["ids2"], batch["len2"])
        loss = info_nce(z1, z2)
        loss.backward()
        opt_pt.step()
        running += loss.item() * batch["ids1"].size(0)
    ep_loss = running / len(pre_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(ep_loss)
    print(f"Pretrain epoch {ep}/{pretrain_epochs} loss={ep_loss:.4f}")

# -------------------- fine-tuning ---------------------------------
ft_epochs = 3
for ep in range(1, ft_epochs + 1):
    # train
    clf.train()
    run = 0
    for batch in train_loader:
        opt_ft.zero_grad()
        logits = clf(batch["ids"], batch["len"])
        loss = ce_loss(logits, batch["label"])
        loss.backward()
        opt_ft.step()
        run += loss.item() * batch["ids"].size(0)
    train_loss = run / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    # val
    clf.eval()
    vloss = 0
    all_p = []
    all_t = []
    all_s = []
    with torch.no_grad():
        for batch in dev_loader:
            logits = clf(batch["ids"], batch["len"])
            loss = ce_loss(logits, batch["label"])
            vloss += loss.item() * batch["ids"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_p.extend(preds)
            all_t.extend(batch["label"].cpu().tolist())
            all_s.extend(batch["seq"])
    vloss /= len(dev_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vloss)
    swa = shape_weighted_accuracy(all_s, all_t, all_p)
    cwa = color_weighted_accuracy(all_s, all_t, all_p)
    comp = complexity_weighted_accuracy(all_s, all_t, all_p)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "CompWA": comp}
    )
    print(
        f"Epoch {ep}: validation_loss = {vloss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} CompWA={comp:.3f}"
    )

# store final predictions / gt
experiment_data["SPR_BENCH"]["predictions"] = all_p
experiment_data["SPR_BENCH"]["ground_truth"] = all_t

# -------------------- plot & save ----------------------------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Fine-tune loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "ft_loss_curve.png"))
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All data saved to", working_dir)

# free memory
del encoder, clf, opt_ft, opt_pt
torch.cuda.empty_cache()
gc.collect()
