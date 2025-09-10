import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────── dirs / logging dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "contrastive_loss": [],
    }
}

# ────────────────────── device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ────────────────────── helper metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def structural_complexity(seq: str) -> int:
    return count_shape_variety(seq) + count_color_variety(seq)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def sc_weighted_accuracy(seqs, y_true, y_pred):
    w = [structural_complexity(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ────────────────────── dataset loading (real or synthetic)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH")
except Exception as e:
    print("Could not load real SPR_BENCH, generating synthetic toy data.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:6])
        colors = list("123456")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seq = " ".join(toks)
            majority_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            seqs.append(seq)
            labels.append(majority_shape)
        ids = list(range(n))
        import datasets

        return datasets.Dataset.from_dict(
            {"id": ids, "sequence": seqs, "label": labels}
        )

    import datasets

    spr = datasets.DatasetDict(
        {"train": synth_split(4000), "dev": synth_split(600), "test": synth_split(600)}
    )

# ────────────────────── vocab
PAD_ID = 0
token2id, label2id = {}, {}
for seq, lab in zip(spr["train"]["sequence"], spr["train"]["label"]):
    for tok in seq.split():
        token2id.setdefault(tok, len(token2id) + 1)
    label2id.setdefault(lab, len(label2id))
MASK_ID = len(token2id) + 1
vocab_size = MASK_ID + 1
id2label = {v: k for k, v in label2id.items()}


def encode_seq(seq):
    return [token2id[tok] for tok in seq.split()]


def encode_lab(l):
    return label2id[l]


# ────────────────────── torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(encode_lab(self.lab[idx]), dtype=torch.long),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    raw = []
    for i, itm in enumerate(batch):
        l = len(itm["input_ids"])
        ids[i, :l] = itm["input_ids"]
        labels[i] = itm["label"]
        raw.append(itm["raw_seq"])
    return {"input_ids": ids, "labels": labels, "raw_seq": raw}


train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ────────────────────── model
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=128, hid_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.out_dim = hid_dim * 2

    def forward(self, ids):
        emb = self.embed(ids)
        lens = (ids != PAD_ID).sum(1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (2,B,H)
        h = torch.cat([h[0], h[1]], dim=1)  # B × 2H
        return h


class SPRModel(nn.Module):
    def __init__(self, vocab, num_labels, emb_dim=128, hid_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = Encoder(vocab, emb_dim, hid_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.cls = nn.Linear(self.encoder.out_dim, num_labels)

    def forward(self, ids, mode="cls"):
        feats = self.encoder(ids)
        if mode == "proj":
            return self.proj(feats)
        if mode == "cls":
            return self.cls(feats)


model = SPRModel(vocab_size, len(label2id)).to(device)


# ────────────────────── augmentations for contrastive
def augment_ids(ids, p_mask=0.15, p_swap=0.1):
    ids = ids.clone()
    B, L = ids.shape
    for b in range(B):
        length = (ids[b] != PAD_ID).sum().item()
        for i in range(length):
            if random.random() < p_mask:
                ids[b, i] = MASK_ID
            elif random.random() < p_swap and i + 1 < length:
                ids[b, i], ids[b, i + 1] = ids[b, i + 1], ids[b, i]
    return ids


# ────────────────────── contrastive loss NT-Xent
def info_nce(z1, z2, temp=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B×D
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp  # 2B×2B
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    pos_sim = sim[torch.arange(2 * B), pos_idx]
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


# ────────────────────── contrastive pre-training
contrast_epochs = 1
opt_c = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1, contrast_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        v1 = augment_ids(ids)
        v2 = augment_ids(ids)
        z1 = model(v1, mode="proj")
        z2 = model(v2, mode="proj")
        loss = info_nce(z1, z2)
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        running_loss += loss.item() * ids.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    experiment_data["SPR_BENCH"]["contrastive_loss"].append(epoch_loss)
    print(f"Contrastive Epoch {epoch}: loss={epoch_loss:.4f}")

# ────────────────────── supervised fine-tuning
criterion = nn.CrossEntropyLoss()
opt_s = torch.optim.Adam(model.parameters(), lr=5e-4)
supervised_epochs = 3
for epoch in range(1, supervised_epochs + 1):
    # train
    model.train()
    tr_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], mode="cls")
        loss = criterion(logits, batch["labels"])
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()
        tr_loss += loss.item() * batch["input_ids"].size(0)
    tr_loss /= len(train_loader.dataset)

    # validate
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], mode="cls")
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["input_ids"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            labs = batch["labels"].cpu().tolist()
            y_pred.extend([id2label[p] for p in preds])
            y_true.extend([id2label[l] for l in labs])
            seqs.extend(batch["raw_seq"])
    val_loss /= len(dev_loader.dataset)

    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    scwa = sc_weighted_accuracy(seqs, y_true, y_pred)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scwa)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCWA={scwa:.3f}"
    )

# ────────────────────── save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
