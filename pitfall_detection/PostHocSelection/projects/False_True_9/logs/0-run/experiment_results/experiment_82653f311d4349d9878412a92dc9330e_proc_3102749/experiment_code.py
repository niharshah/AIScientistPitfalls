import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------------- working dir & device -----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- locate SPR_BENCH ---------------------------
def find_spr_bench() -> pathlib.Path:
    for p in [
        os.environ.get("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and pathlib.Path(p).joinpath("train.csv").exists():
            return pathlib.Path(p).resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ----------------------- metric helpers -----------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def dawa(swa, cwa):  # Dual-Aspect Weighted Accuracy
    return (swa + cwa) / 2


# ----------------------- load dataset -------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)

# ----------------------- vocabulary ---------------------------------
all_tokens = {tok for ex in spr["train"] for tok in ex["sequence"].split()}
MASK_TOKEN = "[MASK]"
all_tokens.add(MASK_TOKEN)
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
MASK_ID = token2id[MASK_TOKEN]
vocab_size = len(token2id) + 1
print(f"Vocab size = {vocab_size}")


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))


# ----------------------- datasets -----------------------------------
class SPRDataset(Dataset):
    def __init__(self, split):
        self.raw = split["sequence"]
        self.enc = [encode(s) for s in self.raw]
        self.label = split["label"] if "label" in split.column_names else None

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = {"input_ids": self.enc[idx], "raw_seq": self.raw[idx]}
        if self.label is not None:
            item["label"] = self.label[idx]
        return item


def pad_sequence(ids, maxlen):
    pad = [PAD_ID] * (maxlen - len(ids))
    return ids + pad


# ---- collate for contrastive pre-training (two views) --------------
def augment(seq_ids):
    ids = seq_ids.copy()
    # token mask 15%
    for i in range(len(ids)):
        if random.random() < 0.15:
            ids[i] = MASK_ID
    # small shuffle
    if random.random() < 0.30 and len(ids) > 3:
        start = random.randint(0, len(ids) - 3)
        end = start + random.randint(2, 3)
        sub = ids[start:end]
        random.shuffle(sub)
        ids[start:end] = sub
    return ids


def collate_contrastive(batch):
    # produce two views per sample
    views = []
    for b in batch:
        v1 = augment(b["input_ids"])
        v2 = augment(b["input_ids"])
        views.append(v1)
        views.append(v2)
    maxlen = max(len(v) for v in views)
    padded = [pad_sequence(v, maxlen) for v in views]
    return torch.tensor(padded, dtype=torch.long)


# ---- collate for supervised training/eval --------------------------
def collate_supervised(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = [pad_sequence(b["input_ids"], maxlen) for b in batch]
    labels = [b["label"] for b in batch]
    raws = [b["raw_seq"] for b in batch]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
        "raw_seq": raws,
    }


train_unlab_loader = DataLoader(
    SPRDataset(spr["train"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_sup_loader = DataLoader(
    SPRDataset(spr["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_supervised,
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_supervised
)


# ----------------------- encoder model ------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden * 2)

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, mask, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, hidden*2]
        return F.normalize(self.proj(h), dim=1)


# ----------------------- pre-training -------------------------------
def info_nce_loss(z, temp=0.07):
    B = z.size(0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp
    sim = sim - torch.eye(B, device=z.device) * 1e9  # remove self-similarity
    pos_idx = torch.arange(0, B, device=z.device) ^ 1  # 0<->1,2<->3 ...
    return F.cross_entropy(sim, pos_idx)


encoder = Encoder(vocab_size).to(device)
opt_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3)

contrastive_epochs = 2
for ep in range(1, contrastive_epochs + 1):
    encoder.train()
    tot, nb = 0, 0
    for batch in train_unlab_loader:
        batch = batch.to(device)
        opt_enc.zero_grad()
        z = encoder(batch)
        loss = info_nce_loss(z)
        loss.backward()
        opt_enc.step()
        tot += loss.item()
        nb += 1
    print(f"Contrastive epoch {ep}: loss={tot/nb:.4f}")


# ----------------------- classifier fine-tuning ---------------------
class SPRClassifier(nn.Module):
    def __init__(self, enc, num_cls):
        super().__init__()
        self.encoder = enc
        self.cls_head = nn.Linear(enc.proj.out_features, num_cls)

    def forward(self, x):
        return self.cls_head(self.encoder(x))


model = SPRClassifier(encoder, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


def evaluate(loader):
    model.eval()
    preds, labels, seqs, loss_sum, n = [], [], [], 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss_sum += loss.item()
            n += 1
            p = logits.argmax(-1).cpu().tolist()
            preds.extend(p)
            labels.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(seqs, labels, preds)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    return loss_sum / n, swa, cwa, dawa(swa, cwa), preds, labels


supervised_epochs = 3
for ep in range(1, supervised_epochs + 1):
    # ---- train ----
    model.train()
    tloss, nb = 0, 0
    for batch in train_sup_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["label"])
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        nb += 1
    tr_loss = tloss / nb
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    # ---- validate ----
    val_loss, swa, cwa, dawa_score, preds, labels = evaluate(dev_loader)
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, swa, cwa, dawa_score))
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f}, SWA={swa:.4f}, CWA={cwa:.4f}, DAWA={dawa_score:.4f}"
    )

# save final predictions / ground truth
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
