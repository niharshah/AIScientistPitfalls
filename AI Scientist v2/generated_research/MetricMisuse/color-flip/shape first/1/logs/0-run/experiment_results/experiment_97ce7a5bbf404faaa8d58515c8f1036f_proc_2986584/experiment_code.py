import os, pathlib, random, math, time, json
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------------------
# -------- helper functions copied from original script -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for s in ["train", "dev", "test"]:
        out[s] = _load(f"{s}.csv")
    return out


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------------------------------------------------------------
# ------------- Load or synth data ---------------------------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print("Loaded SPR_BENCH from", p)
        break
if spr_bench is None:
    print("Dataset not found, making synthetic data.")

    def synth(n):
        seqs, labels = [], []
        shapes, colors = "ABCD", "1234"
        for _ in range(n):
            L = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# ------------------------------------------------------------------
# ------------- Vocabulary & tokenisation --------------------------
def tokenize(s):
    return s.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
mask_id = vocab["<MASK>"]
print("Vocab size:", vocab_size)


# ------------------------------------------------------------------
# ----------------- PyTorch datasets -------------------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, dset):
        self.seqs = dset["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, toks, p=0.15):
        out = [t for t in toks if random.random() >= p]
        if not out:
            out = [random.choice(toks)]
        return out

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class SPRClassifierDataset(Dataset):
    def __init__(self, dset):
        self.seqs = dset["sequence"]
        self.labels = dset["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def encode(lst):
        ids = [torch.tensor([vocab[t] for t in toks]) for toks in lst]
        lens = [len(i) for i in ids]
        return pad_sequence(ids, batch_first=True), torch.tensor(lens)

    i1, l1 = encode(v1)
    i2, l2 = encode(v2)
    return {
        "ids1": i1.to(device),
        "len1": l1.to(device),
        "ids2": i2.to(device),
        "len2": l2.to(device),
    }


def collate_classifier(batch):
    toks, lbl, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in t]) for t in toks]
    lens = [len(i) for i in ids]
    return {
        "ids": pad_sequence(ids, batch_first=True).to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(lbl).to(device),
        "sequence": raw,
    }


pretrain_loader = DataLoader(
    SPRContrastiveDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)


# ------------------------------------------------------------------
# --------------------- Model definitions --------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, hid)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# ------------------------------------------------------------------
def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2N,D
    sim = torch.matmul(z, z.T) / temp  # 2N,2N
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    pos12 = sim[:N, N:]
    pos21 = sim[N:, :N]
    denom12 = torch.logsumexp(sim[:N], 1)
    denom21 = torch.logsumexp(sim[N:], 1)
    loss = (-pos12.diag() + denom12 - pos21.diag() + denom21).mean() * 0.5
    return loss


# ------------------------------------------------------------------
# -------------------- Hyperparameter sweep ------------------------
experiment_data = {"pretrain_epochs": {}}
pretrain_values = [2, 5, 10]
ft_epochs = 3

for pe in pretrain_values:
    tag = f"epochs_{pe}"
    experiment_data["pretrain_epochs"][tag] = {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCHM": []},
        "predictions": [],
        "ground_truth": [],
    }
    # fresh models
    encoder = Encoder(len(vocab)).to(device)
    clf = Classifier(encoder, num_classes=len(set(spr_bench["train"]["label"]))).to(
        device
    )
    # -------- pretrain ----------
    opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for epoch in range(1, pe + 1):
        encoder.train()
        total = 0.0
        for batch in pretrain_loader:
            opt_pt.zero_grad()
            z1 = encoder(batch["ids1"], batch["len1"])
            z2 = encoder(batch["ids2"], batch["len2"])
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt_pt.step()
            total += loss.item() * batch["ids1"].size(0)
        avg = total / len(pretrain_loader.dataset)
        experiment_data["pretrain_epochs"][tag]["losses"]["pretrain"].append(avg)
        print(f"[{tag}] Pretrain epoch {epoch}/{pe} loss={avg:.4f}")
    # -------- fine-tune ---------
    opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, ft_epochs + 1):
        clf.train()
        tot = 0.0
        for batch in train_loader:
            opt_ft.zero_grad()
            logits = clf(batch["ids"], batch["len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            opt_ft.step()
            tot += loss.item() * batch["ids"].size(0)
        train_loss = tot / len(train_loader.dataset)
        experiment_data["pretrain_epochs"][tag]["losses"]["train"].append(train_loss)
        # ---- eval ----
        clf.eval()
        vtot = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                logits = clf(batch["ids"], batch["len"])
                loss = criterion(logits, batch["label"])
                vtot += loss.item() * batch["ids"].size(0)
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(batch["label"].cpu().tolist())
                seqs.extend(batch["sequence"])
        val_loss = vtot / len(dev_loader.dataset)
        experiment_data["pretrain_epochs"][tag]["losses"]["val"].append(val_loss)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        md = experiment_data["pretrain_epochs"][tag]["metrics"]
        md["SWA"].append(swa)
        md["CWA"].append(cwa)
        md["SCHM"].append(schm)
        print(
            f"[{tag}] FT {epoch}/{ft_epochs} train={train_loss:.4f} val={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
        )
    # store final predictions / gts
    experiment_data["pretrain_epochs"][tag]["predictions"] = preds
    experiment_data["pretrain_epochs"][tag]["ground_truth"] = gts
    # ---- plot losses for this setting ----
    plt.figure()
    plt.plot(experiment_data["pretrain_epochs"][tag]["losses"]["train"], label="train")
    plt.plot(experiment_data["pretrain_epochs"][tag]["losses"]["val"], label="val")
    plt.title(f"Fine-tune loss (pretrain={pe})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{tag}.png"))
    plt.close()

# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All results saved in", working_dir)
