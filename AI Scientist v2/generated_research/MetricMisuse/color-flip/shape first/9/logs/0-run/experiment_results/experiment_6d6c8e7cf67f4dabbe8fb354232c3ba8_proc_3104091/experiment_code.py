import os, pathlib, random, numpy as np, torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict as HFDD

# ---------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- dataset helpers ------------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = HFDD()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


def count_shape_variety(seq):  # first char of token encodes shape
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):  # second char encodes colour
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# --------------- load SPR --------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

# ---------------- vocab / labels -------------
vocab = {"<pad>": 0, "<unk>": 1}
for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in enumerate(labels)}
num_classes = len(labels)
print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")


# ------------- augmentation ------------------
def augment(tokens, mask_p=0.15, drop_p=0.10):
    out = []
    for tok in tokens:
        if random.random() < drop_p:  # drop token
            continue
        if random.random() < mask_p:  # mask token
            out.append("<unk>")
        else:
            out.append(tok)
    return out if out else tokens  # avoid empty


# ------------- datasets ----------------------
class ContrastiveDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = [seq.strip().split() for seq in sequences]

    def __len__(self):
        return len(self.seqs)

    def encode(self, toks):
        return [vocab.get(t, vocab["<unk>"]) for t in toks]

    def __getitem__(self, idx):
        toks = self.seqs[idx]
        view1 = self.encode(augment(toks))
        view2 = self.encode(augment(toks))
        return view1, view2


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def pad(seq_list):
        lens = [len(s) for s in seq_list]
        maxlen = max(lens)
        arr = np.full((len(seq_list), maxlen), vocab["<pad>"], np.int64)
        for i, s in enumerate(seq_list):
            arr[i, : len(s)] = s
        return torch.tensor(arr), torch.tensor(lens)

    x1, l1 = pad(v1)
    x2, l2 = pad(v2)
    return (x1.to(device), l1.to(device), x2.to(device), l2.to(device))


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()]

    def __getitem__(self, idx):
        token_ids = self.encode(self.seqs[idx])
        return token_ids, self.labels[idx], self.seqs[idx]


def collate_supervised(batch):
    seqs, labels, raw = zip(*batch)
    lens = [len(s) for s in seqs]
    maxlen = max(lens)
    arr = np.full((len(seqs), maxlen), vocab["<pad>"], np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return (
        torch.tensor(arr).to(device),
        torch.tensor(lens).to(device),
        torch.tensor(labels).to(device),
        raw,
    )


# ---------------- model ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x, lengths):
        emb = self.emb(x)  # B,L,D
        mask = (x != 0).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        return summed / denom  # B,D


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Classifier(nn.Module):
    def __init__(self, encoder, emb_dim, nclass):
        super().__init__()
        self.encoder = encoder
        self.clf = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, nclass)
        )

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        return self.clf(z)


# --------------- loss ------------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    # z1,z2 : (B,D) embeddings
    z = torch.cat([z1, z2], dim=0)  # 2B,D
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature  # 2B,2B
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=device)
    positives = torch.cat([labels + batch_size, labels])
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -9e15)
    loss = F.cross_entropy(sim, positives)
    return loss


# ------------- experiment data ---------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------- contrastive pre-training -------
def pretrain_contrastive(epochs=5, emb_dim=128, lr=3e-4):
    enc = Encoder(vocab_size, emb_dim).to(device)
    proj = ProjectionHead(emb_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=lr)
    train_loader = DataLoader(
        ContrastiveDataset(spr["train"]["sequence"]),
        batch_size=256,
        shuffle=True,
        collate_fn=collate_contrastive,
        drop_last=True,
    )
    enc.train()
    proj.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for x1, l1, x2, l2 in train_loader:
            z1 = proj(enc(x1, l1))
            z2 = proj(enc(x2, l2))
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"[Pretrain] Epoch {ep}: loss={epoch_loss/len(train_loader):.4f}")
    return enc  # return encoder weights


# ---------- fine-tune supervised --------------
def fine_tune(encoder, max_epochs=20, patience=3, lr=1e-3):
    model = Classifier(
        encoder, emb_dim=encoder.emb.embedding_dim, nclass=num_classes
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_supervised,
    )
    val_loader = DataLoader(
        SPRTorchDataset(spr["dev"]),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_supervised,
    )
    best_val, wait = 1e9, 0
    for ep in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        tloss = 0.0
        for x, lens, y, _ in train_loader:
            opt.zero_grad()
            out = model(x, lens)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tloss += loss.item() * x.size(0)
        tloss /= len(train_loader.dataset)
        # ---- val -----
        model.eval()
        vloss = 0.0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for x, lens, y, raw in val_loader:
                out = model(x, lens)
                loss = crit(out, y)
                vloss += loss.item() * x.size(0)
                preds.extend(out.argmax(1).cpu().tolist())
                gts.extend(y.cpu().tolist())
                seqs.extend(raw)
        vloss /= len(val_loader.dataset)
        compwa = complexity_weighted_accuracy(
            seqs, [id2label[i] for i in gts], [id2label[i] for i in preds]
        )
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tloss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(vloss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(compwa)
        print(f"Epoch {ep}: validation_loss = {vloss:.4f}  CompWA = {compwa:.4f}")
        # early stopping
        if vloss < best_val - 1e-4:
            best_val = vloss
            wait = 0
            best_state = model.state_dict()
            best_pred, best_gt, best_seq = preds[:], gts[:], seqs[:]
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    experiment_data["SPR_BENCH"]["predictions"] = best_pred
    experiment_data["SPR_BENCH"]["ground_truth"] = best_gt
    return model


# ---------------- run experiment --------------
encoder = pretrain_contrastive()
model = fine_tune(encoder)

# ------------- save & finish -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Experiment data saved to ./working/")
