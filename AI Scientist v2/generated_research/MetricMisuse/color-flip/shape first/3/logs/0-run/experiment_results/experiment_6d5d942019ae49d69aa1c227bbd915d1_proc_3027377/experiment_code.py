import os, random, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------------------------------------------------------------------
# reproducibility + device
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------------------
# helper metrics
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] if len(tok) > 1 else "#" for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


# -------------------------------------------------------------------------------
# dataset loader (real or synthetic)
def load_spr(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for f in ["train.csv", "dev.csv", "test.csv"]:
        d[f.split(".")[0]] = _ld(f)
    return d


def make_synth(n):
    shapes = list("ABCDE")
    colors = list("12345")
    seqs = []
    labels = []
    for _ in range(n):
        L = random.randint(4, 10)
        s = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(s)
        labels.append(int(count_shape_variety(s) >= count_color_variety(s)))
    return {"sequence": seqs, "label": labels}


SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    raw = load_spr(SPR_PATH)
    print("Loaded real SPR_BENCH")
    train_raw, dev_raw, test_raw = raw["train"], raw["dev"], raw["test"]
except Exception as e:
    print("Falling back to synthetic SPR set:", e)
    train_raw, dev_raw, test_raw = [make_synth(n) for n in (8000, 1500, 1500)]

# -------------------------------------------------------------------------------
# vocab building: separate shape and color
shapes = set(tok[0] for tok in train_raw["sequence"])
colors = set((tok[1] if len(tok) > 1 else "#") for tok in train_raw["sequence"])
shape2id = {s: i + 3 for i, s in enumerate(sorted(shapes))}
color2id = {c: i + 3 for i, c in enumerate(sorted(colors))}
shape2id.update({"<pad>": 0, "<cls>": 1, "<unk>": 2})
color2id.update({"<pad>": 0, "<cls>": 1, "<unk>": 2})
pad_sid, cls_sid, unk_sid = 0, 1, 2
pad_cid, cls_cid, unk_cid = 0, 1, 2

max_len = max(len(seq.split()) for seq in train_raw["sequence"]) + 1  # +cls


def encode(seq: str, max_len=max_len):
    s_ids = [cls_sid]
    c_ids = [cls_cid]
    for tok in seq.split():
        s_ids.append(shape2id.get(tok[0], unk_sid))
        c_ids.append(color2id.get(tok[1] if len(tok) > 1 else "#", unk_cid))
    s_ids = s_ids[:max_len] + [pad_sid] * max(0, max_len - len(s_ids))
    c_ids = c_ids[:max_len] + [pad_cid] * max(0, max_len - len(c_ids))
    return s_ids, c_ids


# -------------------------------------------------------------------------------
# augmentations (context-aware)
def augment(seq: str):
    toks = seq.split()
    op = random.choice(["mask", "shape_shuffle", "color_shuffle", "identity"])
    if op == "mask":
        for i in range(len(toks)):
            if random.random() < 0.15:
                toks[i] = "X#"  # placeholder token
    elif op == "shape_shuffle" and len(toks) > 1:
        random.shuffle(toks)
    elif op == "color_shuffle":
        colors = [tok[1] if len(tok) > 1 else "#" for tok in toks]
        random.shuffle(colors)
        toks = [tok[0] + c for tok, c in zip(toks, colors)]
    return " ".join(toks)


# -------------------------------------------------------------------------------
# torch datasets
class ContrastiveSPR(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        v1 = encode(augment(self.seqs[idx]))
        v2 = encode(augment(self.seqs[idx]))
        return {
            "s1": torch.tensor(v1[0]),
            "c1": torch.tensor(v1[1]),
            "s2": torch.tensor(v2[0]),
            "c2": torch.tensor(v2[1]),
        }


class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s_ids, c_ids = encode(self.seqs[idx])
        return {
            "s": torch.tensor(s_ids),
            "c": torch.tensor(c_ids),
            "y": torch.tensor(self.labels[idx]),
            "raw": self.seqs[idx],
        }


# -------------------------------------------------------------------------------
# model: shape+color embeddings -> summed -> Transformer -> CLS
class ContextEncoder(nn.Module):
    def __init__(self, shape_vocab, color_vocab, emb=64, n_layers=2, n_heads=4):
        super().__init__()
        self.shape_emb = nn.Embedding(len(shape_vocab), emb, padding_idx=pad_sid)
        self.color_emb = nn.Embedding(len(color_vocab), emb, padding_idx=pad_cid)
        self.pos_emb = nn.Embedding(max_len, emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=n_heads, dim_feedforward=emb * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, s_ids, c_ids):
        B, L = s_ids.shape
        pos = torch.arange(L, device=s_ids.device).unsqueeze(0).expand(B, L)
        x = self.shape_emb(s_ids) + self.color_emb(c_ids) + self.pos_emb(pos)
        mask = s_ids == pad_sid
        h = self.transformer(x, src_key_padding_mask=mask)  # (B,L,emb)
        return h[:, 0]  # CLS


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class SPRClassifier(nn.Module):
    def __init__(self, encoder, in_dim, n_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, s_ids, c_ids):
        h = self.encoder(s_ids, c_ids)
        return self.fc(h)


# -------------------------------------------------------------------------------
# loss: InfoNCE
def info_nce(z, temperature=0.1):
    B = z.size(0) // 2
    z = nn.functional.normalize(z, dim=-1)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])
    sim.fill_diagonal_(-9e15)
    return nn.CrossEntropyLoss()(sim, labels)


# -------------------------------------------------------------------------------
# build data loaders
subset = 5000 if isinstance(train_raw, dict) else min(5000, len(train_raw))
contrast_ds = ContrastiveSPR(
    train_raw["sequence"][:subset]
    if isinstance(train_raw, dict)
    else train_raw["sequence"][:subset]
)
contra_loader = DataLoader(contrast_ds, batch_size=256, shuffle=True, drop_last=True)

train_ds = SPRDataset(train_raw["sequence"], train_raw["label"])
dev_ds = SPRDataset(dev_raw["sequence"], dev_raw["label"])
test_ds = SPRDataset(test_raw["sequence"], test_raw["label"])
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -------------------------------------------------------------------------------
# bookkeeping
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"val_acc": [], "val_swa": [], "val_cwa": []},
        "losses": {"train": [], "val": []},
        "aca": {"val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------------------------------------------------------------------
# initialise models
emb_dim = 64
encoder = ContextEncoder(shape2id, color2id, emb=emb_dim).to(device)
proj = ProjectionHead(emb_dim, 128).to(device)
opt_c = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)

# -------------------------------------------------------------------------------
# 1. contrastive pre-training
for epoch in range(1, 4):
    encoder.train()
    proj.train()
    tot, n = 0, 0
    for batch in contra_loader:
        s1 = batch["s1"].to(device)
        c1 = batch["c1"].to(device)
        s2 = batch["s2"].to(device)
        c2 = batch["c2"].to(device)
        z1 = proj(encoder(s1, c1))
        z2 = proj(encoder(s2, c2))
        loss = info_nce(torch.cat([z1, z2], 0))
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        bs = s1.size(0)
        tot += loss.item() * bs
        n += bs
    print(f"Contrastive Epoch {epoch}: loss={tot/n:.4f}")

# -------------------------------------------------------------------------------
# 2. fine-tuning
clf = SPRClassifier(encoder, emb_dim, len(set(train_raw["label"]))).to(device)
opt_f = torch.optim.Adam(clf.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

best_val = -1
best_state = None


def eval_model(model, loader, compute_aca=False, M=3):
    model.eval()
    tot, n = 0, 0
    preds, gts, raws = [], [], []
    with torch.no_grad():
        for b in loader:
            s = b["s"].to(device)
            c = b["c"].to(device)
            y = b["y"].to(device)
            logits = model(s, c)
            loss = criterion(logits, y)
            tot += loss.item() * y.size(0)
            n += y.size(0)
            p = logits.argmax(1).cpu().tolist()
            preds += p
            gts += y.cpu().tolist()
            raws += b["raw"]
    loss = tot / n
    acc = sum(p == g for p, g in zip(preds, gts)) / len(preds)
    swa = shape_weighted_accuracy(raws, gts, preds)
    cwa = color_weighted_accuracy(raws, gts, preds)
    aca = None
    if compute_aca:
        correct = 0
        total = 0
        for raw, g in zip(raws, gts):
            for _ in range(M):
                aug = augment(raw)
                s_ids, c_ids = encode(aug)
                pred = (
                    model(
                        torch.tensor(s_ids).unsqueeze(0).to(device),
                        torch.tensor(c_ids).unsqueeze(0).to(device),
                    )
                    .argmax(1)
                    .item()
                )
                correct += int(pred == g)
                total += 1
        aca = correct / total
    return loss, acc, swa, cwa, aca, preds, gts


for epoch in range(1, 4):
    clf.train()
    tot, n = 0, 0
    for b in train_loader:
        s = b["s"].to(device)
        c = b["c"].to(device)
        y = b["y"].to(device)
        logits = clf(s, c)
        loss = criterion(logits, y)
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = tot / n
    val_loss, val_acc, val_swa, val_cwa, val_aca, _, _ = eval_model(
        clf, dev_loader, compute_aca=True
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | acc={val_acc:.4f} | ACA={val_aca:.4f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_cwa"].append(val_cwa)
    experiment_data["SPR_BENCH"]["aca"]["val"].append(val_aca)
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.cpu() for k, v in clf.state_dict().items()}

# -------------------------------------------------------------------------------
# 3. test evaluation
clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})
test_loss, test_acc, test_swa, test_cwa, test_aca, test_pred, test_gt = eval_model(
    clf, test_loader, compute_aca=True
)
print(
    f"Test: loss={test_loss:.4f} | acc={test_acc:.4f} | SWA={test_swa:.4f} | CWA={test_cwa:.4f} | ACA={test_aca:.4f}"
)

experiment_data["SPR_BENCH"]["predictions"] = test_pred
experiment_data["SPR_BENCH"]["ground_truth"] = test_gt
experiment_data["SPR_BENCH"]["aca"]["test"] = test_aca
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
