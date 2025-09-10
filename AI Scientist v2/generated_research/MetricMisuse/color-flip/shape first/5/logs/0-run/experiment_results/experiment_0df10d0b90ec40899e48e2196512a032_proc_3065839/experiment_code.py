import os, random, math, time, pathlib, csv
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# Utilities required by benchmark
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def harmonic_sca(swa, cwa, eps=1e-8):
    return 2 * swa * cwa / (swa + cwa + eps)


# ------------------------------------------------------------------
# Data – use local SPR_BENCH if present, else create synthetic small one
def generate_synthetic(path: pathlib.Path):
    shapes = ["A", "B", "C", "D"]
    colors = ["1", "2", "3"]

    def gen_seq():
        tokens = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(5, 10))
        ]
        return " ".join(tokens)

    def gen_csv(filename, n):
        with open(path / filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                seq = gen_seq()
                label = int(count_shape_variety(seq) % 2 == 0)  # simple rule
                w.writerow([i, seq, label])

    gen_csv("train.csv", 2000)
    gen_csv("dev.csv", 500)
    gen_csv("test.csv", 500)


def load_csv_dataset(folder: pathlib.Path) -> Dict[str, List[Dict]]:
    data = {}
    for split in ["train", "dev", "test"]:
        with open(folder / f"{split}.csv") as f:
            rdr = csv.DictReader(f)
            data[split] = [row for row in rdr]
            for r in data[split]:
                r["label"] = int(r["label"])
    return data


DATA_PATH = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
if not DATA_PATH.exists():
    os.makedirs(DATA_PATH, exist_ok=True)
    generate_synthetic(DATA_PATH)
datasets = load_csv_dataset(DATA_PATH)
print({k: len(v) for k, v in datasets.items()})

# ------------------------------------------------------------------
# Vocabulary
PAD, MASK = "<PAD>", "<MASK>"


def build_vocab(samples):
    vocab = {PAD: 0, MASK: 1}
    idx = 2
    for s in samples:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab([r["sequence"] for r in datasets["train"]])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str) -> List[int]:
    return [vocab[tok] for tok in seq.split()]


# ------------------------------------------------------------------
# Dataset objects
class SPRDataset(Dataset):
    def __init__(self, rows, supervised=True):
        self.rows = rows
        self.supervised = supervised

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        seq_ids = encode(row["sequence"])
        if self.supervised:
            return {"input": seq_ids, "label": row["label"], "seq": row["sequence"]}
        else:
            return {"input": seq_ids, "seq": row["sequence"]}


def collate(batch):
    maxlen = max(len(b["input"]) for b in batch)
    inp = []
    labels = []
    seqs = []
    for b in batch:
        seqs.append(b["seq"])
        pad = b["input"] + [0] * (maxlen - len(b["input"]))
        inp.append(pad)
        if "label" in b:
            labels.append(b["label"])
    inp = torch.tensor(inp, dtype=torch.long)
    out = {"input": inp, "seq": seqs}
    if labels:
        out["label"] = torch.tensor(labels, dtype=torch.long)
    return out


# ------------------------------------------------------------------
# Augmentation for contrastive learning
def augment(ids: List[int]) -> List[int]:
    new = []
    for tok in ids:
        r = random.random()
        if r < 0.1:
            continue  # deletion
        if r < 0.2:
            new.append(1)  # mask token id=1
        else:
            new.append(tok)
    if len(new) == 0:
        new = ids
    return new


# ------------------------------------------------------------------
# Model
class Encoder(nn.Module):
    def __init__(self, vocab, dim=128, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)  # [B,L,E]
        lengths = (x != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return h[-1]  # [B,H]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, enc_out, n_cls):
        super().__init__()
        self.fc = nn.Linear(enc_out, n_cls)

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------------------
# Contrastive loss (NT-Xent)
def nt_xent(z, t=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / t  # [2B,2B]
    B = z.size(0) // 2
    labels = torch.arange(B, device=z.device)
    loss = 0.0
    for i in range(B):
        pos = sim[i, i + B]
        denom = torch.cat([sim[i, :i], sim[i, i + 1 :]])
        loss += -torch.log(torch.exp(pos) / (torch.exp(denom).sum() + 1e-8))
        j = i + B
        pos = sim[j, i]
        denom = torch.cat([sim[j, :j], sim[j, j + 1 :]])
        loss += -torch.log(torch.exp(pos) / (torch.exp(denom).sum() + 1e-8))
    return loss / (2 * B)


# ------------------------------------------------------------------
# Prepare loaders
batch_size = 128
contrast_loader = DataLoader(
    SPRDataset(datasets["train"], supervised=False),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
train_loader = DataLoader(
    SPRDataset(datasets["train"], supervised=True),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(datasets["dev"], supervised=True),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRDataset(datasets["test"], supervised=True),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ------------------------------------------------------------------
# Training utils
def evaluate(model_enc, model_clf, loader):
    model_enc.eval()
    model_clf.eval()
    y_true, y_pred, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            logits = model_clf(model_enc(x))
            pred = logits.argmax(1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(batch["label"].tolist())
            seqs.extend(batch["seq"])
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hsca = harmonic_sca(swa, cwa)
    return swa, cwa, hsca


# ------------------------------------------------------------------
# Experiment data storage
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------
# Instantiate models
encoder = Encoder(vocab_size).to(device)
proj = ProjectionHead().to(device)
optim_enc = torch.optim.Adam(
    list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
)

# ------------------------------------------------------------------
# Stage 1: Contrastive pre-training
epochs_pre = 3
for epoch in range(1, epochs_pre + 1):
    encoder.train()
    proj.train()
    tot_loss = 0
    cnt = 0
    for batch in contrast_loader:
        ids = batch["input"]
        views1 = [augment(seq.tolist()) for seq in ids]
        views2 = [augment(seq.tolist()) for seq in ids]

        # encode and pad
        def to_tensor(seqs):
            maxlen = max(len(s) for s in seqs)
            arr = [s + [0] * (maxlen - len(s)) for s in seqs]
            return torch.tensor(arr, dtype=torch.long)

        v1 = to_tensor(views1).to(device)
        v2 = to_tensor(views2).to(device)
        z1 = proj(encoder(v1))
        z2 = proj(encoder(v2))
        loss = nt_xent(torch.cat([z1, z2], dim=0))
        optim_enc.zero_grad()
        loss.backward()
        optim_enc.step()
        tot_loss += loss.item()
        cnt += 1
    print(f"[Contrastive] Epoch {epoch}: loss={tot_loss/cnt:.4f}")

# ------------------------------------------------------------------
# Stage 2: Supervised fine-tuning
classifier = Classifier(128, 2).to(device)
optim_all = torch.optim.Adam(
    list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3
)
criterion = nn.CrossEntropyLoss()

epochs_sup = 5
for epoch in range(1, epochs_sup + 1):
    encoder.train()
    classifier.train()
    tr_loss = 0
    tr_cnt = 0
    for batch in train_loader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        logits = classifier(encoder(x))
        loss = criterion(logits, y)
        optim_all.zero_grad()
        loss.backward()
        optim_all.step()
        tr_loss += loss.item()
        tr_cnt += 1
    val_swa, val_cwa, val_hsca = evaluate(encoder, classifier, dev_loader)
    print(
        f"Epoch {epoch}: validation_loss = {tr_loss/tr_cnt:.4f} | HSCA={val_hsca:.4f}"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(val_hsca)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss / tr_cnt)

# ------------------------------------------------------------------
# Final evaluation on test split
test_swa, test_cwa, test_hsca = evaluate(encoder, classifier, test_loader)
print(f"TEST -> SWA: {test_swa:.4f} | CWA: {test_cwa:.4f} | HSCA: {test_hsca:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["val"].append(test_hsca)
experiment_data["SPR_BENCH"]["predictions"] = []  # placeholder
experiment_data["SPR_BENCH"]["ground_truth"] = []  # placeholder

# ------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
