# Bag-of-Tokens ablation of the SPR baseline -------------------------------
# (self-contained single-file programme)

import os, random, math, pathlib, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# mandatory working dir + gpu setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------
# SPR dataset (real or synthetic fallback)
def safe_load_spr():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("SPR_BENCH")
        if DATA_PATH.exists():
            return load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Could not load real SPR_BENCH, falling back to synthetic.", e)

    shapes, colors = list("ABCDEFG"), list("123456")

    def random_token():
        return random.choice(shapes) + random.choice(colors)

    def random_seq():
        return " ".join(random_token() for _ in range(random.randint(4, 10)))

    def label_fn(seq):
        return 1 if sum(tok[0] in "ABC" for tok in seq.split()) % 2 == 0 else 0

    data = {"train": [], "dev": [], "test": []}
    for split, n in [("train", 2000), ("dev", 400), ("test", 400)]:
        for i in range(n):
            seq = random_seq()
            data[split].append({"id": i, "sequence": seq, "label": label_fn(seq)})
    return data


dset = safe_load_spr()
print({k: len(v) for k, v in dset.items()})


# --------------------------------------------------------------------------
# metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def cwca(seqs, y_true, y_pred):
    w = [(count_shape_variety(s) + count_color_variety(s)) / 2 for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# --------------------------------------------------------------------------
# vocab + tokenisation
PAD, MASK, UNK = "<PAD>", "<MASK>", "<UNK>"


def build_vocab(sequences):
    vocab = set(itertools.chain.from_iterable(s.split() for s in sequences))
    vocab = [PAD, MASK, UNK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


vocab, stoi = build_vocab([r["sequence"] for r in dset["train"]])
itos = {i: s for s, i in stoi.items()}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, stoi[UNK]) for tok in seq.split()]


# --------------------------------------------------------------------------
# datasets / dataloaders
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


# contrastive augmentation
def augment(ids: List[int]) -> List[int]:
    ids = ids.copy()
    for i in range(len(ids)):
        if random.random() < 0.15:
            ids[i] = stoi[MASK]
    for i in range(len(ids) - 1):
        if random.random() < 0.1:
            ids[i], ids[i + 1] = ids[i + 1], ids[i]
    if len(ids) > 4 and random.random() < 0.3:
        del ids[random.randint(0, len(ids) - 1)]
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


# --------------------------------------------------------------------------
# Bag-of-Tokens encoder (ablation)
class BagOfTokensEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.linear = nn.Linear(d_model, d_model)  # optional projection

    def forward(self, x, lengths):
        emb = self.emb(x)  # (B,L,D)
        mask = (x != 0).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)  # (B,D)
        mean = summed / ((mask.sum(1)) + 1e-6)
        return self.linear(mean)  # (B,D)


# projection head and full model
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
        self.encoder = BagOfTokensEncoder(vocab_size)
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x, lengths):
        rep = self.encoder(x, lengths)
        logits = self.classifier(rep)
        return logits, rep


# --------------------------------------------------------------------------
# contrastive loss (NT-Xent as in SimCLR)
def nt_xent(z, temperature=0.5):
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    sim = torch.exp(torch.matmul(z, z.T) / temperature)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim * mask
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], 0)
    denom = sim.sum(1)
    loss = -torch.log(pos / denom)
    return loss.mean()


# --------------------------------------------------------------------------
# experiment logging skeleton
experiment_data = {
    "BagOfTokensEncoder": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# --------------------------------------------------------------------------
# build model & optimisers
model = SPRModel(len(vocab), num_labels=len(set(r["label"] for r in dset["train"]))).to(
    device
)
proj_head = ProjectionHead(128).to(device)
opt_contrastive = torch.optim.Adam(
    list(model.encoder.parameters()) + list(proj_head.parameters()), lr=1e-3
)

# --------------------------------------------------------------------------
# 1) contrastive pre-training
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
    print(f"[Pretrain] Epoch {epoch}/{epochs_ct} loss={np.mean(losses):.4f}")

# --------------------------------------------------------------------------
# 2) supervised fine-tuning
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def run_epoch(loader, train=True):
    (model.train() if train else model.eval())
    total_loss, preds, trues, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            logits, rep = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            p = logits.argmax(1).detach().cpu().numpy()
            t = batch["label"].detach().cpu().numpy()
            preds.extend(p)
            trues.extend(t)
            seqs.extend(
                [
                    " ".join(itos[idx.item()] for idx in row if idx.item() != 0)
                    for row in batch["input_ids"].cpu()
                ]
            )
    cwca_val = cwca(seqs, trues, preds)
    return total_loss / len(loader.dataset), cwca_val, preds, trues


best_val = 0
clf_epochs = 5
for epoch in range(1, clf_epochs + 1):
    train_loss, train_cwca, _, _ = run_epoch(train_loader, True)
    val_loss, val_cwca, _, _ = run_epoch(dev_loader, False)
    experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )
    experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["metrics"]["train"].append(
        train_cwca
    )
    experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["metrics"]["val"].append(
        val_cwca
    )
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CWCA={val_cwca:.4f}"
    )
    if val_cwca > best_val:
        best_val = val_cwca

# --------------------------------------------------------------------------
# 3) final test
test_loss, test_cwca, preds, trues = run_epoch(test_loader, False)
print(f"Test CWCA = {test_cwca:.4f}")
experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["metrics"]["test"].append(test_cwca)
experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["predictions"] = preds
experiment_data["BagOfTokensEncoder"]["SPR_BENCH"]["ground_truth"] = trues

# --------------------------------------------------------------------------
# 4) t-SNE visualisation
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
plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap="tab10", s=10)
plt.title("t-SNE of dev embeddings (Bag-of-Tokens)")
plt.savefig(os.path.join(working_dir, "tsne_dev.png"))

# --------------------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data and plot to ./working/")
