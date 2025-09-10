import os, pathlib, random, math, time, gc, json
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

# ---------- I/O / misc ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load (or synthesize) SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


data_root_candidates = [pathlib.Path(p) for p in ["SPR_BENCH", "./data/SPR_BENCH"]]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print("Loaded dataset from", p.resolve())
        break

if spr_bench is None:
    print("SPR_BENCH not found – generating synthetic fallback.")
    shapes, colors = "ABCD", "1234"

    def synth(n):
        seq, lab = [], []
        for i in range(n):
            L = random.randint(4, 9)
            seq.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            lab.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seq, "label": lab}

    spr_bench = DatasetDict()
    for split, n in [("train", 600), ("dev", 150), ("test", 150)]:
        spr_bench[split] = HFDataset.from_dict(synth(n))


# ------------- helpers -------------
def tokenize(s: str):
    return s.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
mask_id = vocab["<MASK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------- metrics -------------
def _shape_var(s):
    return len({t[0] for t in tokenize(s)})


def _color_var(s):
    return len({t[1] for t in tokenize(s)})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_shape_var(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [_color_var(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def seq_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [_shape_var(s) * _color_var(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ------------- datasets -------------
class ContrastiveNoAugDataset(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return toks, toks


class ClassifierDataset(Dataset):
    def __init__(self, hf):
        self.seqs, self.lab = hf["sequence"], hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.lab[idx], self.seqs[idx]


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def enc(tok_lists):
        ids = [
            torch.tensor([vocab[t] for t in toks], dtype=torch.long)
            for toks in tok_lists
        ]
        lens = [len(t) for t in ids]
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        return ids, torch.tensor(lens, dtype=torch.long)

    ids1, l1 = enc(v1)
    ids2, l2 = enc(v2)
    return {"ids1": ids1, "len1": l1, "ids2": ids2, "len2": l2}


def collate_classifier(batch):
    toks, labs, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in tk], dtype=torch.long) for tk in toks]
    lens = [len(i) for i in ids]
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return {
        "ids": ids,
        "len": torch.tensor(lens, dtype=torch.long),
        "label": torch.tensor(labs, dtype=torch.long),
        "sequence": raw,
    }


# ------------- model -------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, n_cls)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    sim.fill_diagonal_(-9e15)
    logits_12 = sim[:N, N:]
    logits_21 = sim[N:, :N]
    loss = (
        -(logits_12.diag())
        + torch.logsumexp(sim[:N], 1)
        + -(logits_21.diag())
        + torch.logsumexp(sim[N:], 1)
    ).mean() * 0.5
    return loss


# ------------- experiment -------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

pretrain_epochs = 4
ft_epochs = 3

pretrain_loader = DataLoader(
    ContrastiveNoAugDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    ClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    ClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)

encoder = Encoder(vocab_size).to(device)
clf = Classifier(encoder, len(set(spr_bench["train"]["label"]))).to(device)

# -------- pre-train ----------
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
for ep in range(1, pretrain_epochs + 1):
    encoder.train()
    tot = 0.0
    for batch in pretrain_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt_pt.zero_grad()
        z1 = encoder(batch["ids1"], batch["len1"])
        z2 = encoder(batch["ids2"], batch["len2"])
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        opt_pt.step()
        tot += loss.item() * batch["ids1"].size(0)
    tot /= len(pretrain_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(tot)
    print(f"Pretrain epoch {ep}/{pretrain_epochs}: loss={tot:.4f}")

# -------- fine-tune ----------
opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for ep in range(1, ft_epochs + 1):
    clf.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt_ft.zero_grad()
        out = clf(batch["ids"], batch["len"])
        loss = criterion(out, batch["label"])
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item() * batch["ids"].size(0)
    tr_loss /= len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # ----- validation -----
    clf.eval()
    val_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            seqs.extend(batch["sequence"])
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = clf(batch["ids"], batch["len"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["ids"].size(0)
            preds += logits.argmax(1).cpu().tolist()
            gts += batch["label"].cpu().tolist()
    val_loss /= len(dev_loader.dataset)

    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    scwa = seq_complexity_weighted_accuracy(seqs, gts, preds)

    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scwa)
    experiment_data["SPR_BENCH"]["epochs"].append(ep)

    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCWA={scwa:.3f}"
    )

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# -------- plotting ----------
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.xlabel("Fine-tune epoch")
plt.ylabel("Loss")
plt.title("FT loss curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_ft.png"))
plt.close()

# -------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy and plot to ./working/")
