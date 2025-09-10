import os, random, math, pathlib, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# mandatory working dir + gpu setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# Try SPR_BENCH, else synthetic data
def safe_load_spr():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("SPR_BENCH")
        if DATA_PATH.exists():
            dset = load_spr_bench(DATA_PATH)
            return dset
    except Exception as e:
        print("Could not load real SPR_BENCH, falling back to synthetic data.", e)

    # synthetic toy dataset -------------------------------------------------
    shapes = list("ABCDEFG")
    colors = list("123456")

    def random_token():
        return random.choice(shapes) + random.choice(colors)

    def random_seq():
        return " ".join(random_token() for _ in range(random.randint(4, 10)))

    def label_fn(seq):
        return 1 if sum(tok[0] in "ABC" for tok in seq.split()) % 2 == 0 else 0

    synthetic = {"train": [], "dev": [], "test": []}
    for split, n in [("train", 2000), ("dev", 400), ("test", 400)]:
        for i in range(n):
            seq = random_seq()
            synthetic[split].append({"id": i, "sequence": seq, "label": label_fn(seq)})
    return synthetic


dset = safe_load_spr()
print({k: len(v) for k, v in dset.items()})


# ------------------------------------------------------------
# utility metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def cwca(seqs, y_true, y_pred):
    weights = [(count_shape_variety(s) + count_color_variety(s)) / 2 for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ------------------------------------------------------------
# vocab + tokenisation
PAD, MASK, UNK = "<PAD>", "<MASK>", "<UNK>"


def build_vocab(sequences):
    vocab = set(itertools.chain.from_iterable(seq.split() for seq in sequences))
    vocab = [PAD, MASK, UNK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


vocab, stoi = build_vocab([ex["sequence"] for ex in dset["train"]])
itos = {i: s for s, i in stoi.items()}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, stoi[UNK]) for tok in seq.split()]


# ------------------------------------------------------------
# datasets
class SPRDataset(Dataset):
    def __init__(self, records, with_label=True):
        self.records = records
        self.with_label = with_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        out = {"input_ids": torch.tensor(encode(rec["sequence"]), dtype=torch.long)}
        if self.with_label:
            out["label"] = torch.tensor(rec["label"], dtype=torch.long)
        return out


def collate_classification(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return {
        "input_ids": padded.to(device),
        "label": labels.to(device),
        "lengths": torch.tensor(lengths).to(device),
    }


# ------------------------------------------------------------
# contrastive augmentation
def augment(ids: List[int]) -> List[int]:
    ids = ids.copy()
    # random masking
    for i in range(len(ids)):
        if random.random() < 0.15:
            ids[i] = stoi[MASK]
    # local shuffle
    for i in range(len(ids) - 1):
        if random.random() < 0.1:
            ids[i], ids[i + 1] = ids[i + 1], ids[i]
    # random dropout
    if len(ids) > 4 and random.random() < 0.3:
        drop_idx = random.randint(0, len(ids) - 1)
        del ids[drop_idx]
    return ids


def collate_contrastive(batch):
    base = [b["input_ids"] for b in batch]
    views = []
    for ids in base:
        views.append(torch.tensor(augment(ids.tolist()), dtype=torch.long))
        views.append(torch.tensor(augment(ids.tolist()), dtype=torch.long))
    lengths = [len(v) for v in views]
    max_len = max(lengths)
    padded = torch.zeros(len(views), max_len, dtype=torch.long)
    for i, v in enumerate(views):
        padded[i, : len(v)] = v
    return {"input_ids": padded.to(device), "lengths": torch.tensor(lengths).to(device)}


# ------------------------------------------------------------
# model
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, lengths):
        emb = self.emb(x)  # (B,L,D)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        h_mean = (h * mask).sum(1) / (mask.sum(1) + 1e-6)
        return self.linear(h_mean)


class ProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SPRModel(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x, lengths):
        rep = self.encoder(x, lengths)
        logits = self.classifier(rep)
        return logits, rep


# ------------------------------------------------------------
# contrastive loss (SimCLR style)
def nt_xent(z, temperature=0.5):
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    sim = torch.exp(torch.matmul(z, z.T) / temperature)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim * mask
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
    denom = sim.sum(dim=1)
    loss = -torch.log(pos / denom)
    return loss.mean()


# ------------------------------------------------------------
# data loaders
batch_c = 128
train_contrastive_loader = DataLoader(
    SPRDataset(dset["train"], with_label=False),
    batch_size=batch_c // 2,
    shuffle=True,
    collate_fn=collate_contrastive,
    drop_last=True,
)
train_loader = DataLoader(
    SPRDataset(dset["train"]),
    batch_size=64,
    shuffle=True,
    collate_fn=collate_classification,
)
dev_loader = DataLoader(
    SPRDataset(dset["dev"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)
test_loader = DataLoader(
    SPRDataset(dset["test"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)

# ------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------
# build model / optim
model = SPRModel(len(vocab), num_labels=len(set(r["label"] for r in dset["train"]))).to(
    device
)
proj_head = ProjectionHead(128).to(device)
opt_contrastive = torch.optim.Adam(
    list(model.encoder.parameters()) + list(proj_head.parameters()), lr=1e-3
)

# ------------------------------------------------------------
# 1. quick contrastive pretraining
epochs_ct = 3
for epoch in range(1, epochs_ct + 1):
    model.train()
    proj_head.train()
    losses = []
    for batch in train_contrastive_loader:
        reps = model.encoder(batch["input_ids"], batch["lengths"])
        feats = proj_head(reps)
        loss = nt_xent(feats)
        opt_contrastive.zero_grad()
        loss.backward()
        opt_contrastive.step()
        losses.append(loss.item())
    print(f"[Pretrain] Epoch {epoch}/{epochs_ct} - loss={np.mean(losses):.4f}")

# ------------------------------------------------------------
# 2. classification fine-tune
clf_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def run_epoch(loader, train=True):
    (model.train() if train else model.eval())
    total_loss, preds, trues, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            logits, _ = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            pred = logits.argmax(1).detach().cpu().numpy()
            true = batch["label"].detach().cpu().numpy()
            preds.extend(pred)
            trues.extend(true)
            seqs.extend(
                [
                    " ".join(itos[idx.item()] for idx in row if idx.item() != 0)
                    for row in batch["input_ids"].cpu()
                ]
            )
    cwca_val = cwca(seqs, trues, preds)
    return total_loss / len(loader.dataset), cwca_val, preds, trues


best_val = 0
for epoch in range(1, clf_epochs + 1):
    train_loss, train_cwca, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_cwca, _, _ = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_cwca)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_cwca)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_CWCA={val_cwca:.4f}"
    )
    if val_cwca > best_val:
        best_val = val_cwca

# ------------------------------------------------------------
# 3. final test evaluation
test_loss, test_cwca, preds, trues = run_epoch(test_loader, train=False)
print(f"Test CWCA = {test_cwca:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test"].append(test_cwca)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = trues

# ------------------------------------------------------------
# 4. t-SNE visualisation on dev set
model.eval()
embeddings, labels = [], []
with torch.no_grad():
    for batch in dev_loader:
        _, reps = model(batch["input_ids"], batch["lengths"])
        embeddings.append(reps.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
embeddings = np.concatenate(embeddings, 0)
tsne = TSNE(n_components=2, init="random", perplexity=30, random_state=0).fit_transform(
    embeddings
)
plt.figure(figsize=(6, 5))
scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap="tab10", s=10)
plt.title("t-SNE of dev embeddings")
plt.savefig(os.path.join(working_dir, "tsne_dev.png"))

# ------------------------------------------------------------
# save experiment_data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data and plot saved to ./working/")
