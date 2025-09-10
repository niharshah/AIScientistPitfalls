import os, random, math, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import DatasetDict, load_dataset

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- GPU / CPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"contrastive": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- dataset path resolver ----------
def resolve_spr_path() -> pathlib.Path:
    tried = []
    # 1) environment variable
    env_path = os.getenv("SPR_DATA_PATH")
    if env_path:
        p = pathlib.Path(env_path)
        tried.append(p)
        if (p / "train.csv").exists():
            return p

    # 2) common absolute location (from provided doc)
    hard = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    tried.append(hard)
    if (hard / "train.csv").exists():
        return hard

    # 3) walk up to search for SPR_BENCH
    cur = pathlib.Path.cwd()
    for ancestor in [cur] + list(cur.parents):
        candidate = ancestor / "SPR_BENCH"
        tried.append(candidate)
        if (candidate / "train.csv").exists():
            return candidate

    msg = "Unable to locate SPR_BENCH dataset.\nTried:\n" + "\n".join(map(str, tried))
    raise FileNotFoundError(msg)


# ---------- dataset utils (unchanged except path call) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------- hyper-parameters ----------
MAX_LEN, EMB_DIM, HIDDEN_DIM = 40, 128, 128
BATCH_SIZE, CONTRAST_EPOCHS, CLS_EPOCHS = 256, 3, 5
TEMPERATURE, AUG_DEL_P, AUG_SWAP_P = 0.1, 0.1, 0.1
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------- vocabulary ----------
class Vocab:
    PAD, MASK, OOV = 0, 1, 2

    def __init__(self, sequences):
        toks = set()
        for s in sequences:
            toks.update(s.strip().split())
        self.stoi = {"<PAD>": 0, "<MASK>": 1, "<OOV>": 2}
        for tok in sorted(toks):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, seq):
        return [self.stoi.get(tok, self.OOV) for tok in seq.strip().split()]

    def __len__(self):
        return len(self.stoi)


# ---------- augmentations ----------
def augment_ids(ids):
    ids = ids.copy()
    # deletion
    ids = [t for t in ids if random.random() > AUG_DEL_P or len(ids) <= 1]
    # swap
    for i in range(len(ids) - 1):
        if random.random() < AUG_SWAP_P:
            ids[i], ids[i + 1] = ids[i + 1], ids[i]
    return ids


# ---------- dataset classes ----------
class ContrastiveSPR(Dataset):
    def __init__(self, sequences, vocab):
        self.seqs, self.vocab = sequences, vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.seqs[idx])[:MAX_LEN]
        return (
            torch.tensor(augment_ids(ids), dtype=torch.long),
            torch.tensor(augment_ids(ids), dtype=torch.long),
        )


class ClassificationSPR(Dataset):
    def __init__(self, sequences, labels, vocab):
        self.seqs, self.labels, self.vocab = sequences, labels, vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.seqs[idx])[:MAX_LEN]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


def collate_pad(seqs):
    return pad_sequence(seqs, batch_first=True, padding_value=0)


# ---------- models ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.rnn = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        out, _ = self.rnn(self.emb(x))
        out = out * mask
        rep = out.sum(1) / mask.sum(1).clamp(min=1)
        return rep


class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        return nn.functional.normalize(z, dim=-1)


def nt_xent(z1, z2, temp=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp
    eye = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(eye, -9e15)
    targets = torch.arange(B, device=z.device)
    loss = (
        nn.functional.cross_entropy(sim[:B, B:], targets)
        + nn.functional.cross_entropy(sim[B:, :B], targets)
    ) * 0.5
    return loss


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(HIDDEN_DIM * 2, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------- load data ----------
data_root = resolve_spr_path()
print("Located SPR_BENCH at", data_root)
ds = load_spr_bench(data_root)
train_seq, dev_seq, test_seq = (
    ds["train"]["sequence"],
    ds["dev"]["sequence"],
    ds["test"]["sequence"],
)
train_lab, dev_lab, test_lab = (
    ds["train"]["label"],
    ds["dev"]["label"],
    ds["test"]["label"],
)
num_classes = len(set(train_lab))
vocab = Vocab(train_seq)
print("Vocab size:", len(vocab))

# ---------- data loaders ----------
contrast_loader = DataLoader(
    ContrastiveSPR(train_seq, vocab),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda b: {
        "v1": collate_pad([x for x, _ in b]),
        "v2": collate_pad([y for _, y in b]),
    },
)

train_loader = DataLoader(
    ClassificationSPR(train_seq, train_lab, vocab),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: {
        "x": collate_pad([x for x, _ in b]),
        "y": torch.stack([y for _, y in b]),
    },
)
dev_loader = DataLoader(
    ClassificationSPR(dev_seq, dev_lab, vocab),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda b: {
        "x": collate_pad([x for x, _ in b]),
        "y": torch.stack([y for _, y in b]),
    },
)
test_loader = DataLoader(
    ClassificationSPR(test_seq, test_lab, vocab),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda b: {
        "x": collate_pad([x for x, _ in b]),
        "y": torch.stack([y for _, y in b]),
    },
)

# ---------- contrastive pre-training ----------
encoder = Encoder(len(vocab)).to(device)
simclr = SimCLR(encoder).to(device)
optim_c = torch.optim.Adam(simclr.parameters(), lr=1e-3)

print("Contrastive pre-training")
for epoch in range(1, CONTRAST_EPOCHS + 1):
    simclr.train()
    total, batches = 0.0, 0
    for batch in contrast_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = nt_xent(simclr(batch["v1"]), simclr(batch["v2"]), TEMPERATURE)
        optim_c.zero_grad()
        loss.backward()
        optim_c.step()
        total += loss.item()
        batches += 1
    avg = total / batches
    experiment_data["SPR_BENCH"]["losses"]["contrastive"].append((epoch, avg))
    print(f"  epoch {epoch}/{CONTRAST_EPOCHS}  loss={avg:.4f}")

# ---------- supervised fine-tuning ----------
classifier = Classifier(encoder, num_classes).to(device)
optim_f = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    preds, labels, seqs = [], [], []
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x"])
            loss = criterion(out, batch["y"])
            total += loss.item()
            batches += 1
            preds.extend(out.argmax(-1).cpu().numpy())
            labels.extend(batch["y"].cpu().numpy())
            for row in batch["x"].cpu():
                seqs.append(
                    " ".join(
                        vocab.itos[id.item()] for id in row if id.item() not in (0,)
                    )
                )
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
    return total / batches, swa, cwa, hwa, preds, labels


print("Supervised fine-tuning")
for epoch in range(1, CLS_EPOCHS + 1):
    classifier.train()
    total, batches = 0.0, 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = criterion(classifier(batch["x"]), batch["y"])
        optim_f.zero_grad()
        loss.backward()
        optim_f.step()
        total += loss.item()
        batches += 1
    train_loss = total / batches
    val_loss, swa, cwa, hwa, _, _ = evaluate(classifier, dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, train_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, swa, cwa, hwa))
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f}  CWA={cwa:.3f}  HWA={hwa:.3f}"
    )

# ---------- test ----------
test_loss, swa, cwa, hwa, preds, labels = evaluate(classifier, test_loader)
experiment_data["SPR_BENCH"]["metrics"]["train"] = []  # placeholder for symmetry
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels
print(f"\nTest  SWA={swa:.3f}  CWA={cwa:.3f}  HWA={hwa:.3f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
