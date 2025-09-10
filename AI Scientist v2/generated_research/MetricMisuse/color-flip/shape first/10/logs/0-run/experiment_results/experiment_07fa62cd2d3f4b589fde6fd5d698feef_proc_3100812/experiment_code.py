import os, pathlib, random, string, math, numpy as np, torch, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────── house-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────── metrics helpers
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def structural_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ─────────────────── load dataset (real or synthetic fallback)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Real dataset unavailable; generating synthetic data.", e)

    def synth_split(n):
        shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seq = " ".join(toks)
            maj_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            seqs.append(seq)
            labels.append(maj_shape)
        ids = list(range(n))
        return {"id": ids, "sequence": seqs, "label": labels}

    import datasets

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(2000)),
            "dev": datasets.Dataset.from_dict(synth_split(400)),
            "test": datasets.Dataset.from_dict(synth_split(400)),
        }
    )

# ─────────────────── vocab encoding
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(ds):
    global token2id, label2id
    toks, labs = set(), set()
    for s, l in zip(ds["sequence"], ds["label"]):
        toks.update(s.split())
        labs.add(l)
    token2id = {tok: i + 1 for i, tok in enumerate(sorted(toks))}
    label2id = {lab: i for i, lab in enumerate(sorted(labs))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_seq(s):
    return [token2id[t] for t in s.split()]


def encode_label(l):
    return label2id[l]


# ─────────────────── datasets
class SPRTorch(Dataset):
    def __init__(self, hf_ds, unlabeled=False):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"] if not unlabeled else [None] * len(self.seq)
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(encode_seq(self.seq[idx]), dtype=torch.long),
            "raw_seq": self.seq[idx],
        }
        if not self.unlabeled:
            item["label_id"] = torch.tensor(
                encode_label(self.lab[idx]), dtype=torch.long
            )
        return item


def pad_collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    out = {"input_ids": ids, "raw_seq": [b["raw_seq"] for b in batch]}
    if "label_id" in batch[0]:
        out["labels"] = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    return out


train_ul_loader = DataLoader(
    SPRTorch(spr["train"], unlabeled=True),
    batch_size=256,
    shuffle=True,
    collate_fn=pad_collate,
)
train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=128, shuffle=True, collate_fn=pad_collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=pad_collate
)


# ─────────────────── model
class Encoder(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.linear = nn.Linear(dim, dim)

    def forward(self, ids):
        emb = self.embed(ids)
        mask = (ids != PAD_ID).unsqueeze(-1)
        pooled = (emb * mask).sum(1) / (mask.sum(1).clamp(min=1))
        z = nn.functional.normalize(self.linear(torch.relu(pooled)), dim=-1)
        return z


class ContrastiveModel(nn.Module):
    def __init__(self, vocab, dim, num_labels):
        super().__init__()
        self.encoder = Encoder(vocab, dim)
        self.cls = nn.Linear(dim, num_labels)

    def forward(self, ids, represent=False):
        z = self.encoder(ids)
        if represent:
            return z
        return self.cls(z)


# ─────────────────── data augmentation for contrastive pairs
def augment_tokens(tok_ids):
    tok = tok_ids.copy()
    # random masking
    for i in range(len(tok)):
        if random.random() < 0.15:
            tok[i] = PAD_ID
    # slight shuffle (swap adjacent with prob)
    for i in range(len(tok) - 1):
        if random.random() < 0.1:
            tok[i], tok[i + 1] = tok[i + 1], tok[i]
    return tok


# ─────────────────── contrastive pretraining
dim = 128
model = ContrastiveModel(len(token2id) + 1, dim, len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
temperature = 0.07
epochs_pre = 5


def nt_xent(z1, z2):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2B x D
    sim = torch.matmul(z, z.T) / temperature  # 2B x 2B
    mask = ~torch.eye(2 * B, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2 * B, -1)  # remove self-similarities
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], 0).to(z.device)
    pos_sim = torch.sum(z * z.roll(B, 0), 1) / temperature
    loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim).sum(1)).mean()
    return loss


for epoch in range(1, epochs_pre + 1):
    model.train()
    total_loss = 0
    c = 0
    for batch in train_ul_loader:
        ids = batch["input_ids"].tolist()
        aug1 = [torch.tensor(augment_tokens(x), dtype=torch.long) for x in ids]
        aug2 = [torch.tensor(augment_tokens(x), dtype=torch.long) for x in ids]
        # pad
        max_len = max(max(len(a) for a in aug1), max(len(a) for a in aug2))
        a1 = torch.zeros(len(ids), max_len, dtype=torch.long)
        a2 = torch.zeros_like(a1)
        for i, (x, y) in enumerate(zip(aug1, aug2)):
            a1[i, : len(x)] = x
            a2[i, : len(y)] = y
        a1, a2 = a1.to(device), a2.to(device)
        z1 = model.encoder(a1)
        z2 = model.encoder(a2)
        loss = nt_xent(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ids)
        c += len(ids)
    avg_loss = total_loss / c
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(avg_loss)
    print(f"Pretrain Epoch {epoch}: contrastive_loss = {avg_loss:.4f}")

# ─────────────────── supervised fine-tune
ft_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, ft_epochs + 1):
    model.train()
    tr_loss = 0
    n = 0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(ids)
        loss = criterion(logits, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(ids)
        n += len(ids)
    tr_loss /= n
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # ─ eval
    model.eval()
    vl_loss = 0
    n = 0
    y_true, y_pred, seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            logits = model(ids)
            loss = criterion(logits, lbl)
            vl_loss += loss.item() * len(ids)
            n += len(ids)
            preds = logits.argmax(1).cpu().tolist()
            labs = lbl.cpu().tolist()
            seqs.extend(batch["raw_seq"])
            y_true.extend([id2label[x] for x in labs])
            y_pred.extend([id2label[x] for x in preds])
    vl_loss /= n
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    scwa = structural_weighted_accuracy(seqs, y_true, y_pred)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scwa)
    print(
        f"Epoch {epoch}: validation_loss = {vl_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCWA={scwa:.3f}"
    )

# store last predictions
experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all metrics to", os.path.join(working_dir, "experiment_data.npy"))
