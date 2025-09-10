import os, pathlib, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -------------------- working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- locate SPR_BENCH ------------------------
def find_spr_bench():
    cands = [
        os.environ.get("SPR_DATA_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cands:
        if p and pathlib.Path(p).joinpath("train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench()


# -------------------- metrics ---------------------------------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0


def dawa(swa_, cwa_):
    return 0.5 * (swa_ + cwa_)


# -------------------- load dataset ----------------------------
def load_spr(root):
    ld = lambda f: load_dataset(
        "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(train=ld("train.csv"), dev=ld("dev.csv"), test=ld("test.csv"))


spr = load_spr(DATA_PATH)

# -------------------- vocabulary ------------------------------
all_tokens = set(tok for seq in spr["train"]["sequence"] for tok in seq.split())
token2id = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
MASK_ID = 1
vocab_size = len(token2id) + 2


def encode(seq):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))


# -------------------- PyTorch datasets ------------------------
class SPRDataset(Dataset):
    def __init__(self, split, with_label=True):
        self.seqs = split["sequence"]
        self.ids = [encode(s) for s in self.seqs]
        self.with_label = with_label
        self.labels = split["label"] if with_label else [0] * len(self.seqs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }
        if self.with_label:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_ids = []
    labels = []
    raw = []
    for b in batch:
        seq = b["input_ids"]
        if maxlen - len(seq):
            seq = torch.cat([seq, torch.full((maxlen - len(seq),), PAD_ID)])
        pad_ids.append(seq)
        if "label" in b:
            labels.append(b["label"])
        raw.append(b["raw_seq"])
    out = {"input_ids": torch.stack(pad_ids), "raw_seq": raw}
    if labels:
        out["label"] = torch.stack(labels)
    return out


# -------------------- data augmentations ----------------------
def augment(x, mask_prob=0.15, swap_prob=0.1):
    x = x.clone()
    for i in range(len(x)):
        if x[i] == PAD_ID:
            break
        if random.random() < mask_prob:
            x[i] = MASK_ID
    if random.random() < swap_prob and len(x[x != PAD_ID]) > 1:
        idx = list(range(len(x[x != PAD_ID])))
        i, j = random.sample(idx, 2)
        x[i], x[j] = x[j].clone(), x[i].clone()
    return x


# -------------------- model definitions -----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_sz, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = x == PAD_ID
        h = self.enc(h, src_key_padding_mask=mask)
        # mean pooling over valid tokens
        lens = (~mask).sum(1, keepdim=True)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / lens.clamp(min=1)
        return F.normalize(pooled, dim=-1)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_cls):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder.tok_emb.embedding_dim, num_cls)

    def forward(self, x):
        z = self.encoder(x)
        return self.cls(z), z


# -------------------- contrastive loss ------------------------
def info_nce(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)  # (2N,D)
    sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)  # 2N x 2N
    N = z1.size(0)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)  # remove self-similarity
    sim /= temp
    loss = F.cross_entropy(sim, labels)
    return loss


# -------------------- loaders ------------------------
batch_size = 256
unlab_loader = DataLoader(
    SPRDataset(spr["train"], with_label=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)

# -------------------- experiment logging ---------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------- pre-training ----------------------
encoder = Encoder(vocab_size).to(device)
opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
epochs_pt = 2
for ep in range(1, epochs_pt + 1):
    encoder.train()
    tot_loss = 0
    nb = 0
    for batch in unlab_loader:
        ids = batch["input_ids"].to(device)
        aug1 = torch.stack([augment(x.clone()) for x in ids]).to(device)
        aug2 = torch.stack([augment(x.clone()) for x in ids]).to(device)
        z1 = encoder(aug1)
        z2 = encoder(aug2)
        loss = info_nce(z1, z2)
        opt_pt.zero_grad()
        loss.backward()
        opt_pt.step()
        tot_loss += loss.item()
        nb += 1
    l = tot_loss / nb
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append((ep, l))
    print(f"Pretrain epoch {ep}: loss={l:.4f}")

# -------------------- fine-tuning -----------------------
model = SPRModel(encoder, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs_ft = 4
for ep in range(1, epochs_ft + 1):
    # ---- train ----
    model.train()
    tot = 0
    nb = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits, _ = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    tr_loss = tot / nb
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    # ---- validate ----
    model.eval()
    tot = 0
    nb = 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, _ = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            tot += loss.item()
            nb += 1
            p = logits.argmax(-1).cpu().tolist()
            l = batch["label"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["raw_seq"])
    v_loss = tot / nb
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, v_loss))
    s = swa(seqs, labels, preds)
    c = cwa(seqs, labels, preds)
    d = dawa(s, c)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, s, c, d))
    print(f"Epoch {ep}: val_loss={v_loss:.4f} SWA={s:.4f} CWA={c:.4f} DAWA={d:.4f}")
    if ep == epochs_ft:
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = labels

# -------------------- save ------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data.")
