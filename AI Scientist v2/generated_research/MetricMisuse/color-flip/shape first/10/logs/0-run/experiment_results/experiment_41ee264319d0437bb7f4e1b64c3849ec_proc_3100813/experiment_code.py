import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────── house-keeping & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "losses": {"contrastive": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ──────────────────────────────── metric helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def structural_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ──────────────────────────────── load SPR dataset (real or synthetic)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real data, generating synthetic.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:6])
        colors = list("123456")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seqs.append(" ".join(toks))
            maj = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            labels.append(maj)
        ids = list(range(n))
        import datasets

        return datasets.Dataset.from_dict(
            {"id": ids, "sequence": seqs, "label": labels}
        )

    import datasets

    spr = datasets.DatasetDict(
        {"train": synth_split(4000), "dev": synth_split(800), "test": synth_split(800)}
    )

# ──────────────────────────────── build vocabularies
PAD_ID = 0
MASK_ID = 1  # special mask token
token2id, label2id = {}, {}


def build_vocabs(ds):
    global token2id, label2id
    tokens, labels = set(), set()
    for seq, lab in zip(ds["sequence"], ds["label"]):
        tokens.update(seq.split())
        labels.add(lab)
    token2id = {tok: idx + 2 for idx, tok in enumerate(sorted(tokens))}  # reserve 0,1
    label2id = {lab: idx for idx, lab in enumerate(sorted(labels))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):
    return [token2id[t] for t in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ──────────────────────────────── torch dataset wrappers
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_sequence(self.seq[idx]), dtype=torch.long),
            "label_id": torch.tensor(encode_label(self.lab[idx]), dtype=torch.long),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    raw = [b["raw_seq"] for b in batch]
    return {"input_ids": ids, "labels": labels, "raw_seq": raw}


train_loader_super = DataLoader(
    SPRTorch(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)


# ──────────────────────────────── data augmentation for contrastive
def augment_ids(ids_tensor):
    ids = ids_tensor.clone()
    # token masking
    mask_prob = 0.15
    mask = (torch.rand_like(ids.float()) < mask_prob) & (ids != PAD_ID)
    ids[mask] = MASK_ID
    # mild shuffle: swap two tokens in 10% of sequences
    for row in ids:
        if random.random() < 0.1:
            non_pad = (row != PAD_ID).nonzero(as_tuple=True)[0]
            if len(non_pad) > 1:
                i, j = random.sample(non_pad.tolist(), 2)
                row[i], row[j] = row[j].clone(), row[i].clone()
    return ids


# ──────────────────────────────── encoder & heads
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=128, proj_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, proj_dim)
        )

    def mean_pool(self, emb, ids):
        mask = (ids != PAD_ID).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        return summed / denom

    def forward(self, ids, project=False):
        emb = self.embedding(ids)
        feat = self.mean_pool(emb, ids)
        return self.proj(feat) if project else feat


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.embedding.embedding_dim, num_labels)

    def forward(self, ids):
        feat = self.encoder(ids, project=False)
        return self.fc(feat)


# ──────────────────────────────── contrastive pre-training
encoder = Encoder(len(token2id) + 2, emb_dim=128, proj_dim=128).to(device)
optim_c = torch.optim.Adam(encoder.parameters(), lr=1e-3)
temp = 0.07
EPOCHS_CONTR = 3

contrast_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
for epoch in range(1, EPOCHS_CONTR + 1):
    encoder.train()
    running_loss, seen = 0.0, 0
    for batch in contrast_loader:
        ids = batch["input_ids"].to(device)
        v1 = augment_ids(ids)
        v2 = augment_ids(ids)
        inp = torch.cat([v1, v2], 0).to(device)
        z = encoder(inp, project=True)
        z = nn.functional.normalize(z, dim=1)
        B = ids.size(0)
        sim = torch.matmul(z, z.T) / temp  # 2B x 2B
        mask = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(mask, -1e9)
        targets = torch.arange(B, device=sim.device)
        logits1 = sim[:B, B:]  # positives are diagonal of this
        logits2 = sim[B:, :B]
        loss = (
            nn.functional.cross_entropy(logits1, targets)
            + nn.functional.cross_entropy(logits2, targets)
        ) / 2
        optim_c.zero_grad()
        loss.backward()
        optim_c.step()
        running_loss += loss.item() * B
        seen += B
    epoch_loss = running_loss / seen
    experiment_data["SPR_BENCH"]["losses"]["contrastive"].append(epoch_loss)
    print(f"Contrastive Epoch {epoch}: loss = {epoch_loss:.4f}")

# ──────────────────────────────── supervised fine-tuning
model = Classifier(encoder, len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS_SUP = 5

for epoch in range(1, EPOCHS_SUP + 1):
    # ---- train
    model.train()
    tr_loss, seen = 0.0, 0
    for batch in train_loader_super:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * ids.size(0)
        seen += ids.size(0)
    tr_loss /= seen
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # ---- validate
    model.eval()
    val_loss, seen = 0.0, 0
    y_true, y_pred, seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            val_loss += loss.item() * ids.size(0)
            seen += ids.size(0)
            preds = logits.argmax(1).cpu().tolist()
            labs = labels.cpu().tolist()
            y_true.extend([id2label[x] for x in labs])
            y_pred.extend([id2label[x] for x in preds])
            seqs.extend(batch["raw_seq"])
    val_loss /= seen
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    scwa = structural_weighted_accuracy(seqs, y_true, y_pred)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scwa)

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCWA={scwa:.3f}"
    )

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true

# ──────────────────────────────── save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
