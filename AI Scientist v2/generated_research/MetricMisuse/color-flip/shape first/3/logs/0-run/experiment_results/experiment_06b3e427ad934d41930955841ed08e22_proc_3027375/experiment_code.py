import os, random, pathlib, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------------
# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -----------------------------------------------------------------------------------
# Reproducibility + device
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------------
# SPR helpers  ----------------------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) + 1e-9)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / (sum(w) + 1e-9)


# -----------------------------------------------------------------------------------
# Load SPR_BENCH or synthetic fallback ---------------------------------------------
def load_spr(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for fn in ["train.csv", "dev.csv", "test.csv"]:
        d[fn.split(".")[0]] = _ld(fn)
    return d


def make_synth(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(4, 10)
        s = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(s)
        labels.append(int(count_shape_variety(s) > count_color_variety(s)))
    return {"sequence": seqs, "label": labels}


SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    raw = load_spr(SPR_PATH)
    print("Loaded real SPR_BENCH.")
    train_raw = raw["train"]
    dev_raw = raw["dev"]
    test_raw = raw["test"]
except Exception as e:
    print("Could not load SPR_BENCH, using synthetic.", e)
    train_raw = make_synth(4000)
    dev_raw = make_synth(800)
    test_raw = make_synth(800)

# -----------------------------------------------------------------------------------
# Vocabulary ------------------------------------------------------------------------
all_tokens = set(tok for seq in train_raw["sequence"] for tok in seq.split())
vocab = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
special_tokens = ["<pad>", "<unk>", "<mask>", "<cls>"]
for i, sp in enumerate(special_tokens):
    vocab[sp] = i
pad_idx, unk_idx, mask_idx, cls_idx = [vocab[s] for s in special_tokens]


def encode(seq, max_len):
    ids = [cls_idx] + [vocab.get(tok, unk_idx) for tok in seq.split()]
    ids = ids[:max_len] + [pad_idx] * (max(0, max_len - len(ids)))
    return ids


max_len = max(len(s.split()) for s in train_raw["sequence"]) + 1  # +cls


# -----------------------------------------------------------------------------------
# Augmentations ---------------------------------------------------------------------
def augment(seq: str):
    toks = seq.split()
    # token masking
    if len(toks) > 2:
        for i in range(len(toks)):
            if random.random() < 0.15:
                toks[i] = "<mask>"
    # light shuffle (swap two tokens)
    if len(toks) > 2 and random.random() < 0.5:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    return " ".join(toks)


# -----------------------------------------------------------------------------------
# Datasets --------------------------------------------------------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, seqs, max_len):
        self.seqs = seqs
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        v1 = torch.tensor(encode(augment(s), self.max_len), dtype=torch.long)
        v2 = torch.tensor(encode(augment(s), self.max_len), dtype=torch.long)
        return {"view1": v1, "view2": v2}


class SPRDataset(Dataset):
    def __init__(self, seqs, labels, max_len):
        self.seqs = seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx], self.max_len), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# -----------------------------------------------------------------------------------
# Model -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            emb_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        em = self.emb(x)  # (B,L,E)
        _, h = self.gru(em)  # (2*n_layers,B,H)
        h = torch.cat([h[-1], h[-2]], dim=-1)  # last fwd & bwd
        return h  # (B,2H)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, h):
        return self.mlp(h)


class Classifier(nn.Module):
    def __init__(self, encoder, in_dim, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


# -----------------------------------------------------------------------------------
# Contrastive Loss ------------------------------------------------------------------
def nt_xent(features, temperature=0.1):
    # features: (2B,D) already L2-normalised
    B = features.shape[0] // 2
    sim = torch.mm(features, features.t()) / temperature
    labels = torch.arange(B, device=features.device)
    labels = torch.cat([labels + B, labels])  # positives indices
    mask = torch.eye(2 * B, device=features.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss


# -----------------------------------------------------------------------------------
# Build datasets / loaders ----------------------------------------------------------
subset_size = min(8000, len(train_raw["sequence"]))  # speed
contrast_ds = ContrastiveDataset(train_raw["sequence"][:subset_size], max_len)
contra_loader = DataLoader(contrast_ds, batch_size=256, shuffle=True, drop_last=True)

train_ds = SPRDataset(train_raw["sequence"], train_raw["label"], max_len)
dev_ds = SPRDataset(dev_raw["sequence"], dev_raw["label"], max_len)
test_ds = SPRDataset(test_raw["sequence"], test_raw["label"], max_len)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -----------------------------------------------------------------------------------
# Book-keeping dict -----------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "aca": {"val": [], "test": []},
    }
}

# -----------------------------------------------------------------------------------
# 1. Contrastive pre-training -------------------------------------------------------
vocab_size = len(vocab)
encoder = Encoder(vocab_size).to(device)
proj = ProjectionHead(in_dim=512).to(device)
opt_c = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)

epochs_c = 5
for epoch in range(1, epochs_c + 1):
    encoder.train()
    proj.train()
    tot_loss = 0
    n = 0
    for batch in contra_loader:
        v1 = batch["view1"].to(device)
        v2 = batch["view2"].to(device)
        f1 = nn.functional.normalize(proj(encoder(v1)), dim=-1)
        f2 = nn.functional.normalize(proj(encoder(v2)), dim=-1)
        feat = torch.cat([f1, f2], 0)
        loss = nt_xent(feat)
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        bs = v1.size(0)
        tot_loss += loss.item() * bs
        n += bs
    print(f"Contrastive Epoch {epoch}: loss={tot_loss/n:.4f}")

# -----------------------------------------------------------------------------------
# 2. Fine-tuning --------------------------------------------------------------------
clf = Classifier(encoder, in_dim=512, n_classes=len(set(train_raw["label"]))).to(device)
opt_f = torch.optim.Adam(clf.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def eval_loop(model, loader, compute_aca=False, M=3):
    model.eval()
    tot_loss = 0
    n = 0
    preds = []
    gts = []
    raws = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = ce(logits, y)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p)
            gts.extend(y.cpu().tolist())
            raws.extend(batch["raw"])
    acc = sum([p == g for p, g in zip(preds, gts)]) / len(preds)
    aca = None
    if compute_aca:
        correct = 0
        total = 0
        for raw_seq, g in zip(raws, gts):
            for _ in range(M):
                aug_seq = augment(raw_seq)
                ids = torch.tensor(encode(aug_seq, max_len)).unsqueeze(0).to(device)
                pred = model(ids).argmax(1).item()
                correct += int(pred == g)
                total += 1
        aca = correct / total
    return tot_loss / n, acc, aca, preds, gts, raws


best_val_acc = -1
best_state = None
epochs_f = 5
for epoch in range(1, epochs_f + 1):
    clf.train()
    tot = 0
    n = 0
    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = clf(x)
        loss = ce(logits, y)
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()
        bs = y.size(0)
        tot += loss.item() * bs
        n += bs
    train_loss = tot / n

    val_loss, val_acc, val_aca, _, _, _ = eval_loop(clf, dev_loader, compute_aca=True)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.4f} | val_aca={val_aca:.4f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["aca"]["val"].append(val_aca)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = clf.state_dict()

# -----------------------------------------------------------------------------------
# 3. Evaluation on test -------------------------------------------------------------
clf.load_state_dict(best_state)
test_loss, test_acc, test_aca, test_pred, test_gt, _ = eval_loop(
    clf, test_loader, compute_aca=True
)
print(f"Test: loss={test_loss:.4f} | acc={test_acc:.4f} | ACA={test_aca:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["train"] = []  # not tracked separately
experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
experiment_data["SPR_BENCH"]["aca"]["test"] = test_aca

# -----------------------------------------------------------------------------------
# Save everything -------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
