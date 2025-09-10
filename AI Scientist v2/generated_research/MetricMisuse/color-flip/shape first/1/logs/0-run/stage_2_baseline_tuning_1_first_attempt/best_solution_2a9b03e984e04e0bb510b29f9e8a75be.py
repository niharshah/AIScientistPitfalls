import os, pathlib, random, math, json, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict

# -------------------  I/O & device ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------  helpers --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# -------------------  dataset prep ---------------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print(f"Loaded data from {p}")
        break
if spr_bench is None:
    print("SPR_BENCH not found – generating synthetic data.")

    def synth(n):
        seqs, labels, shapes, colors = [], [], "ABCD", "1234"
        for _ in range(n):
            L = random.randint(4, 9)
            seqs.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for sp, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[sp] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# -------------------  vocab & tokenisation -------------
def tokenize(s):
    return s.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for t in tokenize(seq):
            if t not in vocab:
                vocab[t] = len(vocab)
vocab_size = len(vocab)
mask_id = vocab["<MASK>"]
print("Vocab size:", vocab_size)


# -------------------  torch datasets -------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, ds):
        self.seqs = ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, toks, p=0.15):
        out = [t for t in toks if random.random() >= p]
        return out if out else [random.choice(toks)]

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class SPRClassifierDataset(Dataset):
    def __init__(self, ds):
        self.seqs, self.labels = ds["sequence"], ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def enc(list_tok):
        ids = [torch.tensor([vocab[t] for t in toks]) for toks in list_tok]
        lens = [len(i) for i in ids]
        return pad_sequence(ids, batch_first=True), torch.tensor(lens)

    ids1, len1 = enc(v1)
    ids2, len2 = enc(v2)
    return {
        "ids1": ids1.to(device),
        "len1": len1.to(device),
        "ids2": ids2.to(device),
        "len2": len2.to(device),
    }


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in t]) for t in toks]
    lens = [len(i) for i in ids]
    return {
        "ids": pad_sequence(ids, batch_first=True).to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(labels).to(device),
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


# -------------------  models --------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, enc, num_classes):
        super().__init__()
        self.encoder = enc
        self.head = nn.Linear(enc.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    sim.masked_fill_(torch.eye(2 * N, dtype=torch.bool, device=sim.device), -9e15)
    logits_12 = sim[:N, N:]
    logits_21 = sim[N:, :N]
    denom_12 = torch.logsumexp(sim[:N], 1)
    denom_21 = torch.logsumexp(sim[N:], 1)
    return (
        (-logits_12.diag() + denom_12) + (-logits_21.diag() + denom_21)
    ).mean() * 0.5


# -------------------  logging dict --------------------
experiment_data = {"fine_tune_epochs": {}}

# -------------------  PRETRAIN encoder ---------------
encoder_base = Encoder(vocab).to(device)
optimizer_pt = torch.optim.Adam(encoder_base.parameters(), lr=1e-3)
pretrain_epochs = 2
for ep in range(1, pretrain_epochs + 1):
    encoder_base.train()
    epoch_loss = 0.0
    for b in pretrain_loader:
        optimizer_pt.zero_grad()
        z1 = encoder_base(b["ids1"], b["len1"])
        z2 = encoder_base(b["ids2"], b["len2"])
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer_pt.step()
        epoch_loss += loss.item() * b["ids1"].size(0)
    print(f"Pretrain Epoch {ep}: loss={epoch_loss/len(pretrain_loader.dataset):.4f}")

pretrain_state = encoder_base.state_dict()  # snapshot to clone later

# -------------------  fine-tune variants -------------
ft_variants = [3, 6, 10, 15]
criterion = nn.CrossEntropyLoss()

for ft_epochs in ft_variants:
    key = f"epochs_{ft_epochs}"
    experiment_data["fine_tune_epochs"][key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCHM": []},
        "predictions": [],
        "ground_truth": [],
        "sequences": [],
    }
    # build fresh encoder/cls starting from pretrain ckpt
    enc = Encoder(vocab).to(device)
    enc.load_state_dict(pretrain_state)
    clf = Classifier(enc, num_classes=len(set(spr_bench["train"]["label"]))).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for ep in range(1, ft_epochs + 1):
        # train
        clf.train()
        run_loss = 0.0
        for b in train_loader:
            opt.zero_grad()
            logits = clf(b["ids"], b["len"])
            loss = criterion(logits, b["label"])
            loss.backward()
            opt.step()
            run_loss += loss.item() * b["ids"].size(0)
        train_loss = run_loss / len(train_loader.dataset)
        experiment_data["fine_tune_epochs"][key]["losses"]["train"].append(train_loss)
        # val
        clf.eval()
        vloss = 0.0
        preds, truth, seqs = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                logits = clf(b["ids"], b["len"])
                loss = criterion(logits, b["label"])
                vloss += loss.item() * b["ids"].size(0)
                p = logits.argmax(1).cpu().tolist()
                preds.extend(p)
                truth.extend(b["label"].cpu().tolist())
                seqs.extend(b["sequence"])
        vloss /= len(dev_loader.dataset)
        swa = shape_weighted_accuracy(seqs, truth, preds)
        cwa = color_weighted_accuracy(seqs, truth, preds)
        schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        ed = experiment_data["fine_tune_epochs"][key]
        ed["losses"]["val"].append(vloss)
        ed["metrics"]["SWA"].append(swa)
        ed["metrics"]["CWA"].append(cwa)
        ed["metrics"]["SCHM"].append(schm)
        if ep == ft_epochs:  # store final predictions
            ed["predictions"] = preds
            ed["ground_truth"] = truth
            ed["sequences"] = seqs
        print(
            f"[{key}] Epoch {ep}/{ft_epochs}  train={train_loss:.4f} val={vloss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
        )

    # optional plot per variant
    plt.figure()
    plt.plot(ed["losses"]["train"], label="train")
    plt.plot(ed["losses"]["val"], label="val")
    plt.title(f"Loss curve ({key})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_{key}.png"))
    plt.close()

# -------------------  save ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all results to", os.path.join(working_dir, "experiment_data.npy"))
