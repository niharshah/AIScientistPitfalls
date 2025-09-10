import os, random, math, pathlib, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# mandatory working dir + gpu setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# 1. build three distinct synthetic datasets -----------------
def make_synthetic(
    shapes,
    colors,
    seq_len_range=(4, 10),
    rule="parity",
    n_train=2000,
    n_dev=400,
    n_test=400,
):
    def random_token():
        return random.choice(shapes) + random.choice(colors)

    def random_seq():
        L = random.randint(*seq_len_range)
        return " ".join(random_token() for _ in range(L))

    if rule == "parity":

        def label_fn(seq):
            return (
                1
                if sum(tok[0] in shapes[: len(shapes) // 2] for tok in seq.split()) % 2
                == 0
                else 0
            )

    elif rule == "color_even":
        tgt = set(colors[: len(colors) // 2])

        def label_fn(seq):
            return 1 if sum(tok[1:] in tgt for tok in seq.split()) % 2 == 0 else 0

    elif rule == "mod3":

        def label_fn(seq):
            return 1 if len(seq.split()) % 3 == 0 else 0

    else:
        raise ValueError(rule)

    out = {sp: [] for sp in ["train", "dev", "test"]}
    for split, n in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        for i in range(n):
            s = random_seq()
            out[split].append({"id": i, "sequence": s, "label": label_fn(s)})
    return out


set1 = make_synthetic(list("ABCDEF"), list("1234"), rule="parity")
set2 = make_synthetic(list("GHIJKL"), list("5678"), rule="color_even")
set3 = make_synthetic(list("MNOPQR"), ["90"], rule="mod3")
datasets = {"Synthetic1": set1, "Synthetic2": set2, "Synthetic3": set3}
print({k: {sp: len(v[sp]) for sp in v} for k, v in datasets.items()})


# ------------------------------------------------------------
# 2. metrics helpers -----------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1:] for tok in sequence.split() if len(tok) > 1))


def cwca(seqs, y_true, y_pred):
    weights = [(count_shape_variety(s) + count_color_variety(s)) / 2 for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------------------------------------------------
# 3. vocab ----------------------------------------------------
PAD, MASK, UNK = "<PAD>", "<MASK>", "<UNK>"


def build_vocab(dsets):
    seqs = list(
        itertools.chain.from_iterable(
            ex["sequence"].split()
            for ds in dsets.values()
            for split in ds.values()
            for ex in split
        )
    )
    vocab = [PAD, MASK, UNK] + sorted(set(seqs))
    return vocab, {tok: i for i, tok in enumerate(vocab)}


vocab, stoi = build_vocab(datasets)
itos = {i: s for s, i in stoi.items()}


def encode(seq: str):
    return [stoi.get(tok, stoi[UNK]) for tok in seq.split()]


# ------------------------------------------------------------
# 4. datasets -------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, records, with_label=True):
        self.data = records
        self.with_label = with_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        out = {"input_ids": torch.tensor(encode(rec["sequence"]), dtype=torch.long)}
        if self.with_label:
            out["label"] = torch.tensor(rec["label"], dtype=torch.long)
        return out


def collate_classification(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor([len(s) for s in seqs])
    max_len = lengths.max()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return {
        "input_ids": padded.to(device),
        "label": labels.to(device),
        "lengths": lengths.to(device),
    }


# contrastive augmentation & collate
def augment(ids):
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
    base_ids = [b["input_ids"] for b in batch]
    view1, view2 = [], []
    for ids in base_ids:
        view1.append(torch.tensor(augment(ids.tolist()), dtype=torch.long))
        view2.append(torch.tensor(augment(ids.tolist()), dtype=torch.long))
    views = view1 + view2  # <-- fixed ordering
    lengths = [len(v) for v in views]
    max_len = max(lengths)
    padded = torch.zeros(len(views), max_len, dtype=torch.long)
    for i, v in enumerate(views):
        padded[i, : len(v)] = v
    return {"input_ids": padded.to(device), "lengths": torch.tensor(lengths).to(device)}


def build_loaders(dset, contrastive=False, batch_size=64):
    if contrastive:
        return DataLoader(
            SPRDataset(dset["train"], with_label=False),
            batch_size=batch_size // 2,
            shuffle=True,
            collate_fn=collate_contrastive,
            drop_last=True,
        )
    tr = DataLoader(
        SPRDataset(dset["train"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_classification,
    )
    dv = DataLoader(
        SPRDataset(dset["dev"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_classification,
    )
    te = DataLoader(
        SPRDataset(dset["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_classification,
    )
    return tr, dv, te


contrastive_loader_1 = build_loaders(set1, True, 128)
train1_loader, dev1_loader, test1_loader = build_loaders(set1)
train2_loader, dev2_loader, test2_loader = build_loaders(set2)
train3_loader, dev3_loader, test3_loader = build_loaders(set3)


# ------------------------------------------------------------
# 5. model ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d, padding_idx=0)
        self.lstm = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(d * 2, d)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        h_mean = (h * mask).sum(1) / (mask.sum(1) + 1e-6)
        return self.lin(h_mean)


class ProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SPRModel(nn.Module):
    def __init__(self, vocab_sz, num_labels=2):
        super().__init__()
        self.encoder = Encoder(vocab_sz)
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x, lengths):
        rep = self.encoder(x, lengths)
        return self.classifier(rep), rep


# contrastive loss
def nt_xent(z, temperature=0.5):
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    sim = torch.exp(torch.mm(z, z.t()) / temperature)
    mask = ~torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim * mask
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], 0)
    denom = sim.sum(1)
    loss = -torch.log(pos / denom)
    return loss.mean()


# ------------------------------------------------------------
# 6. experiment data dict ------------------------------------
experiment_data = {
    "Synthetic1": {
        "metrics": {"train_cwca": [], "val_cwca": [], "train_acr": [], "val_acr": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "Synthetic2": {
        "metrics": {"train_cwca": [], "val_cwca": [], "train_acr": [], "val_acr": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "Synthetic3": {
        "metrics": {"test_cwca": [], "test_acr": []},
        "losses": {"test": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# ------------------------------------------------------------
# 7. helper: ACR computation ---------------------------------
def compute_acr(model, seq_strs, orig_preds, k=3):
    model.eval()
    consistent = 0
    for seq, pr in zip(seq_strs, orig_preds):
        ok = True
        ids = encode(seq)
        for _ in range(k):
            aug_ids = augment(ids.copy())
            t = torch.tensor(aug_ids, dtype=torch.long).unsqueeze(0).to(device)
            l = torch.tensor([t.size(1)]).to(device)
            with torch.no_grad():
                logit, _ = model(t, l)
            if logit.argmax(1).item() != pr:
                ok = False
                break
        consistent += int(ok)
    return consistent / len(seq_strs) if seq_strs else 0.0


# ------------------------------------------------------------
# 8. helper: run epoch ---------------------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train=True, optimizer=None):
    (model.train() if train else model.eval())
    tot_loss, preds, trues, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits, _ = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["label"]) if "label" in batch else None
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if loss is not None:
            tot_loss += loss.item() * batch["label"].size(0)
            t = batch["label"].detach().cpu().numpy()
            p = logits.argmax(1).detach().cpu().numpy()
            trues.extend(t)
            preds.extend(p)
            seqs.extend(
                [
                    " ".join(itos[idx.item()] for idx in row if idx.item() != 0)
                    for row in batch["input_ids"].cpu()
                ]
            )
    cwca_val = cwca(seqs, trues, preds) if preds else 0.0
    avg_loss = tot_loss / len(loader.dataset) if preds else 0.0
    return avg_loss, cwca_val, preds, trues, seqs


# ------------------------------------------------------------
# 9. instantiate models --------------------------------------
model = SPRModel(len(vocab)).to(device)
proj_head = ProjectionHead(128).to(device)
opt_contrastive = torch.optim.Adam(
    list(model.encoder.parameters()) + list(proj_head.parameters()), lr=1e-3
)

# ------------------------------------------------------------
# 10. contrastive pretraining -------------------------------
epochs_ct = 3
for ep in range(1, epochs_ct + 1):
    model.train()
    proj_head.train()
    c_losses = []
    for batch in contrastive_loader_1:
        feats = proj_head(model.encoder(batch["input_ids"], batch["lengths"]))
        loss = nt_xent(feats)
        opt_contrastive.zero_grad()
        loss.backward()
        opt_contrastive.step()
        c_losses.append(loss.item())
    print(f"[Contrastive] Epoch {ep}/{epochs_ct} loss={np.mean(c_losses):.4f}")

# ------------------------------------------------------------
# 11. supervised training on Synthetic1 ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

best_val = 0
clf_epochs = 5
for ep in range(1, clf_epochs + 1):
    tr_loss, tr_cwca, tr_pred, _, tr_seq = run_epoch(train1_loader, True, optimizer)
    acr_train = compute_acr(model, tr_seq, tr_pred)
    val_loss, val_cwca, val_pred, val_true, val_seq = run_epoch(dev1_loader, False)
    acr_val = compute_acr(model, val_seq, val_pred)
    experiment_data["Synthetic1"]["losses"]["train"].append(tr_loss)
    experiment_data["Synthetic1"]["losses"]["val"].append(val_loss)
    experiment_data["Synthetic1"]["metrics"]["train_cwca"].append(tr_cwca)
    experiment_data["Synthetic1"]["metrics"]["val_cwca"].append(val_cwca)
    experiment_data["Synthetic1"]["metrics"]["train_acr"].append(acr_train)
    experiment_data["Synthetic1"]["metrics"]["val_acr"].append(acr_val)
    print(
        f"[S1] Ep{ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_CWCA={val_cwca:.4f} val_ACR={acr_val:.4f}"
    )
    if val_cwca > best_val:
        best_val = val_cwca
        torch.save(model.state_dict(), os.path.join(working_dir, "best_s1.pt"))
experiment_data["Synthetic1"]["predictions"] = val_pred
experiment_data["Synthetic1"]["ground_truth"] = val_true

# ------------------------------------------------------------
# 12. fine-tune on Synthetic2 --------------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_s1.pt")))
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-4)
ft_epochs = 3
for ep in range(1, ft_epochs + 1):
    tr_loss, tr_cwca, tr_pred, _, tr_seq = run_epoch(train2_loader, True, optimizer_ft)
    acr_train = compute_acr(model, tr_seq, tr_pred)
    val_loss, val_cwca, val_pred, val_true, val_seq = run_epoch(dev2_loader, False)
    acr_val = compute_acr(model, val_seq, val_pred)
    experiment_data["Synthetic2"]["losses"]["train"].append(tr_loss)
    experiment_data["Synthetic2"]["losses"]["val"].append(val_loss)
    experiment_data["Synthetic2"]["metrics"]["train_cwca"].append(tr_cwca)
    experiment_data["Synthetic2"]["metrics"]["val_cwca"].append(val_cwca)
    experiment_data["Synthetic2"]["metrics"]["train_acr"].append(acr_train)
    experiment_data["Synthetic2"]["metrics"]["val_acr"].append(acr_val)
    print(
        f"[S2 FT] Ep{ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_CWCA={val_cwca:.4f} val_ACR={acr_val:.4f}"
    )
experiment_data["Synthetic2"]["predictions"] = val_pred
experiment_data["Synthetic2"]["ground_truth"] = val_true

# ------------------------------------------------------------
# 13. final test on Synthetic3 -------------------------------
test_loss, test_cwca, test_pred, test_true, test_seq = run_epoch(test3_loader, False)
test_acr = compute_acr(model, test_seq, test_pred)
experiment_data["Synthetic3"]["losses"]["test"].append(test_loss)
experiment_data["Synthetic3"]["metrics"]["test_cwca"].append(test_cwca)
experiment_data["Synthetic3"]["metrics"]["test_acr"].append(test_acr)
experiment_data["Synthetic3"]["predictions"] = test_pred
experiment_data["Synthetic3"]["ground_truth"] = test_true
print(f"[S3 TEST] loss={test_loss:.4f} CWCA={test_cwca:.4f} ACR={test_acr:.4f}")

# ------------------------------------------------------------
# 14. t-SNE visualisation ------------------------------------
model.eval()
embeds, labels = [], []
with torch.no_grad():
    for batch in dev3_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        _, rep = model(batch["input_ids"], batch["lengths"])
        embeds.append(rep.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
embeds = np.concatenate(embeds, 0)
tsne = TSNE(n_components=2, init="random", perplexity=30, random_state=0).fit_transform(
    embeds
)
plt.figure(figsize=(6, 5))
plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap="tab10", s=10)
plt.title("t-SNE Synthetic3 dev embeddings")
plt.savefig(os.path.join(working_dir, "tsne_s3_dev_fixed.png"))

# ------------------------------------------------------------
# 15. save experiment data -----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All done. Results stored in ./working/")
