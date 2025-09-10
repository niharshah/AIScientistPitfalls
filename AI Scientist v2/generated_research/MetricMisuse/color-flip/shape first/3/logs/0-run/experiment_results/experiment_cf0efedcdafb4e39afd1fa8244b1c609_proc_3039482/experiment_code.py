import os, random, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------#
# Boiler-plate & reproducibility
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# -----------------------------------------------------------------------------#
# Metric helpers
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# -----------------------------------------------------------------------------#
# Load SPR_BENCH (real or synthetic)
def load_real_spr(path):
    try:
        from datasets import load_dataset

        def _ld(csv):
            return load_dataset(
                "csv",
                data_files=str(path / csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {
            "train": _ld("train.csv"),
            "dev": _ld("dev.csv"),
            "test": _ld("test.csv"),
        }
    except Exception as e:
        print("Falling back to synthetic data –", e)
        return False, {}


def make_synth(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) >= count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = load_real_spr(SPR_PATH)
if have_real:
    train_raw, dev_raw, test_raw = raw["train"], raw["dev"], raw["test"]
    train_data = {"sequence": train_raw["sequence"], "label": train_raw["label"]}
    dev_data = {"sequence": dev_raw["sequence"], "label": dev_raw["label"]}
    test_data = {"sequence": test_raw["sequence"], "label": test_raw["label"]}
else:
    train_data, dev_data, test_data = (
        make_synth(6000),
        make_synth(1200),
        make_synth(1200),
    )

# -----------------------------------------------------------------------------#
# Vocabulary & encoding
all_tokens = {tok for seq in train_data["sequence"] for tok in seq.split()}
vocab = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
vocab.update({"<pad>": 0, "<unk>": 1, "<mask>": 2, "<cls>": 3})
inv_vocab = {i: t for t, i in vocab.items()}
pad_id, unk_id, mask_id, cls_id = [
    vocab[t] for t in ["<pad>", "<unk>", "<mask>", "<cls>"]
]
max_len = max(len(s.split()) for s in train_data["sequence"]) + 1


def encode(seq):
    ids = [cls_id] + [vocab.get(t, unk_id) for t in seq.split()]
    ids = ids[:max_len] + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids)


# -----------------------------------------------------------------------------#
# Datasets
class SeqOnlyDS(torch.utils.data.Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


class LabeledDS(torch.utils.data.Dataset):
    def __init__(self, seqs, lbl):
        self.seqs, self.lbl = seqs, lbl

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.lbl[i]


# -----------------------------------------------------------------------------#
# Masking-only augmentation
def augment_mask_only(seq: str):
    toks = [t if random.random() > 0.3 else "<mask>" for t in seq.split()]
    return " ".join(toks)


augment = augment_mask_only  # enforce masking-only policy


# -----------------------------------------------------------------------------#
# Model definitions
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb=128, hid=256, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=pad_id)
        self.gru = nn.GRU(emb, hid, num_layers=layers, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h[-1]


class Classifier(nn.Module):
    def __init__(self, encoder, hid, n_cls):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, x):
        return self.fc(self.enc(x))


def contrastive_loss(z, temp=0.1):
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temp
    N = sim.size(0)
    sim.masked_fill_(torch.eye(N, device=z.device).bool(), -9e15)
    pos = torch.arange(N, device=z.device) ^ 1
    return nn.functional.cross_entropy(sim, pos)


# -----------------------------------------------------------------------------#
# DataLoaders
pretrain_loader = DataLoader(
    SeqOnlyDS(train_data["sequence"]), batch_size=256, shuffle=True, drop_last=True
)
train_loader = DataLoader(
    LabeledDS(train_data["sequence"], train_data["label"]), batch_size=128, shuffle=True
)
dev_loader = DataLoader(
    LabeledDS(dev_data["sequence"], dev_data["label"]), batch_size=256
)
test_loader = DataLoader(
    LabeledDS(test_data["sequence"], test_data["label"]), batch_size=256
)

# -----------------------------------------------------------------------------#
# Experiment bookkeeping
experiment_data = {
    "mask_only": {
        "SPR_BENCH": {
            "losses": {"pretrain": [], "train": [], "val": []},
            "metrics": {"val_acc": [], "val_aca": []},
            "test": {},
        }
    }
}
rec = experiment_data["mask_only"]["SPR_BENCH"]

# -----------------------------------------------------------------------------#
# Contrastive pre-training
enc = Encoder(len(vocab)).to(device)
opt_enc = torch.optim.Adam(enc.parameters(), lr=1e-3)
pre_epochs = 8
print("\n--- Contrastive Pre-training (masking only) ---")
for ep in range(1, pre_epochs + 1):
    enc.train()
    tot = cnt = 0
    t0 = time.time()
    for batch in pretrain_loader:
        views = []
        for s in batch:
            views.append(encode(augment(s)))
            views.append(encode(augment(s)))
        x = torch.stack(views).to(device)
        opt_enc.zero_grad()
        loss = contrastive_loss(enc(x))
        loss.backward()
        opt_enc.step()
        tot += loss.item()
        cnt += 1
    avg = tot / cnt
    rec["losses"]["pretrain"].append(avg)
    print(f"Epoch {ep}: loss={avg:.4f} ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------------#
# Fine-tuning
model = Classifier(enc, hid=256, n_cls=len(set(train_data["label"]))).to(device)
crit = nn.CrossEntropyLoss()
opt_cls = torch.optim.Adam(model.parameters(), lr=1e-3)
f_epochs = 10


def eval_loader(loader):
    model.eval()
    loss_tot = n = 0
    preds = gts = raws = []
    preds = []
    gts = []
    raws = []
    with torch.no_grad():
        for seqs, lbl in loader:
            x = torch.stack([encode(s) for s in seqs]).to(device)
            y = torch.tensor(lbl, device=device)
            logits = model(x)
            loss = crit(logits, y)
            loss_tot += loss.item() * y.size(0)
            n += y.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(lbl)
            raws.extend(seqs)
    return loss_tot / n, np.mean(np.array(preds) == np.array(gts)), preds, gts, raws


def aca_metric(seqs, labels, M=3):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s, l in zip(seqs, labels):
            variants = [s] + [augment(s) for _ in range(M)]
            x = torch.stack([encode(v) for v in variants]).to(device)
            p = model(x).argmax(1).cpu().tolist()
            correct += sum(int(pi == l) for pi in p)
            total += len(variants)
    return correct / total


print("\n--- Fine-tuning ---")
for ep in range(1, f_epochs + 1):
    model.train()
    tot = n = 0
    for seqs, lbl in train_loader:
        x = torch.stack([encode(s) for s in seqs]).to(device)
        y = torch.tensor(lbl, device=device)
        opt_cls.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt_cls.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = tot / n
    val_loss, val_acc, _, _, _ = eval_loader(dev_loader)
    val_aca = aca_metric(dev_data["sequence"], dev_data["label"])
    rec["losses"]["train"].append(train_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["val_acc"].append(val_acc)
    rec["metrics"]["val_aca"].append(val_aca)
    print(
        f"Ep {ep}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | ACA={val_aca:.4f}"
    )

# -----------------------------------------------------------------------------#
# Test evaluation
test_loss, test_acc, test_pred, test_gt, test_raw = eval_loader(test_loader)
test_swa = shape_weighted_accuracy(test_raw, test_gt, test_pred)
test_cwa = color_weighted_accuracy(test_raw, test_gt, test_pred)
test_aca = aca_metric(test_data["sequence"], test_data["label"])
rec["test"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
    "cwa": test_cwa,
    "aca": test_aca,
    "predictions": test_pred,
    "ground_truth": test_gt,
}
print("\n--- Test results ---")
print(
    f"Loss={test_loss:.4f} | Acc={test_acc:.4f} | SWA={test_swa:.4f} | "
    f"CWA={test_cwa:.4f} | ACA={test_aca:.4f}"
)

# -----------------------------------------------------------------------------#
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
