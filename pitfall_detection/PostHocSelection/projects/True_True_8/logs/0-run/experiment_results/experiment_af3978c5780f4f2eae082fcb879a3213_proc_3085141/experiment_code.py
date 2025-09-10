import os, random, string, pathlib, time, math
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# -------------------------- house-keeping --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# -------------------------- load data ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # fallback synthetic tiny data

    def _build(n):
        rows = []
        for _ in range(n):
            l = random.randint(4, 12)
            seq = []
            for _ in range(l):
                seq.append(random.choice("ABCDE") + random.choice("01234"))
            rows.append(
                {
                    "id": str(random.randint(0, 1e9)),
                    "sequence": " ".join(seq),
                    "label": random.randint(0, 1),
                }
            )
        return HFDataset.from_list(rows)

    spr = DatasetDict(train=_build(3000), dev=_build(800), test=_build(800))
print({k: len(v) for k, v in spr.items()})

# -------------------------- vocabulary -----------------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]
V = len(vocab)
MAX_LEN = 40


def encode(seq):
    ids = [vocab.get(t, 1) for t in seq.strip().split()[:MAX_LEN]]
    ids += [pad_idx] * (MAX_LEN - len(ids))
    return ids


# -------------------------- metrics --------------------------------
def count_shape_variety(s):
    return len(set(t[0] for t in s.split()))


def count_color_variety(s):
    return len(set(t[1] for t in s.split() if len(t) > 1))


def swa(seqs, y, g):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w)


def cwa(seqs, y, g):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y, g) if yt == yp) / sum(w)


def compwa(seqs, y, g):
    return swa(seqs, y, g) + cwa(seqs, y, g)


# -------------------------- datasets -------------------------------
class SequenceDS(TorchDataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        return self.hf[i]["sequence"]


class LabeledDS(TorchDataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        row = self.hf[i]
        return row["sequence"], torch.tensor(row["label"], dtype=torch.long)


def collate_unlabeled(batch):
    return batch  # list[str]


def collate_labeled(batch):
    seqs, labels = zip(*batch)
    ids = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    return {"sequence": seqs, "input_ids": ids, "labels": torch.stack(labels)}


# ----------------------- data augmentation -------------------------
def augment(seq):
    toks = seq.split()
    if not toks:
        return seq
    op = random.choice(["mask", "swap", "none"])
    if op == "mask":
        idx = random.randrange(len(toks))
        toks[idx] = UNK
    elif op == "swap" and len(toks) > 1:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return " ".join(toks)


# -------------------------- model ----------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, dim, padding_idx=pad_idx)

    def forward(self, ids):
        emb = self.embed(ids)  # B,L,D
        mask = (ids != pad_idx).unsqueeze(-1).float()
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return pooled  # B,D


class ProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SPRClassifier(nn.Module):
    def __init__(self, encoder, dim, num_cls):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(dim, num_cls)

    def forward(self, ids):
        h = self.encoder(ids)
        return self.fc(h)


# ----------------------- NT-Xent loss --------------------------------
def nt_xent(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    N = z.size(0) // 2
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat(
        [torch.arange(N, device=z.device) + N, torch.arange(N, device=z.device)]
    )
    pos_sim = sim[torch.arange(2 * N, device=z.device), pos]
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


# ----------------------- training params ---------------------------
EMBED_DIM = 128
BATCH_CTR = 256
BATCH_FT = 128
PRE_EPOCHS = 2
FT_EPOCHS = 4
NUM_CLASSES = len(set(spr["train"]["label"]))

# ----------------------- contrastive pre-training ------------------
unl_loader = DataLoader(
    SequenceDS(spr["train"]),
    batch_size=BATCH_CTR,
    shuffle=True,
    collate_fn=collate_unlabeled,
)

encoder = Encoder(V, EMBED_DIM).to(device)
proj = ProjectionHead(EMBED_DIM).to(device)
opt_ct = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=3e-3)

print("\nContrastive pre-training")
for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    proj.train()
    tot_loss = 0
    for seqs in unl_loader:
        views1 = [augment(s) for s in seqs]
        views2 = [augment(s) for s in seqs]
        ids = torch.tensor([encode(s) for s in views1 + views2], dtype=torch.long).to(
            device
        )
        opt_ct.zero_grad()
        h = encoder(ids)
        z = proj(h)
        loss = nt_xent(z)
        loss.backward()
        opt_ct.step()
        tot_loss += loss.item()
    print(f"pre-epoch {ep}: contrastive_loss={tot_loss/len(unl_loader):.4f}")

# ----------------------- fine-tuning --------------------------------
train_loader = DataLoader(
    LabeledDS(spr["train"]),
    batch_size=BATCH_FT,
    shuffle=True,
    collate_fn=collate_labeled,
)
dev_loader = DataLoader(
    LabeledDS(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_labeled
)

classifier = SPRClassifier(encoder, EMBED_DIM, NUM_CLASSES).to(device)
opt_ft = torch.optim.Adam(classifier.parameters(), lr=3e-3)
crit = nn.CrossEntropyLoss()

experiment_data = {
    "contrastive+finetune": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "CompWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}

print("\nSupervised fine-tuning")
for ep in range(1, FT_EPOCHS + 1):
    classifier.train()
    tr_loss = 0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        opt_ft.zero_grad()
        logits = classifier(ids)
        loss = crit(logits, labels)
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)
    experiment_data["contrastive+finetune"]["losses"]["train"].append((ep, tr_loss))

    # ---------- validation ----------
    classifier.eval()
    val_loss, seqs, gt, pr = 0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = classifier(ids)
            val_loss += crit(logits, labels).item()
            preds = logits.argmax(-1).cpu().tolist()
            pr.extend(preds)
            gt.extend(batch["labels"].tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader)
    SWA = swa(seqs, gt, pr)
    CWA = cwa(seqs, gt, pr)
    Comp = SWA + CWA
    experiment_data["contrastive+finetune"]["losses"]["val"].append((ep, val_loss))
    experiment_data["contrastive+finetune"]["metrics"]["SWA"].append((ep, SWA))
    experiment_data["contrastive+finetune"]["metrics"]["CWA"].append((ep, CWA))
    experiment_data["contrastive+finetune"]["metrics"]["CompWA"].append((ep, Comp))
    experiment_data["contrastive+finetune"]["predictions"].append((ep, pr))
    experiment_data["contrastive+finetune"]["ground_truth"].append((ep, gt))
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={SWA:.3f} "
        f"CWA={CWA:.3f} CompWA={Comp:.3f}"
    )

# ------------------------ save -------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
