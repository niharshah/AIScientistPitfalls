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
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


# -----------------------------------------------------------------------------#
# Load SPR_BENCH or synthetic fallback
def load_real_spr(path: pathlib.Path):
    try:
        from datasets import load_dataset

        def _load(csv):
            return load_dataset(
                "csv",
                data_files=str(path / csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
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
    train_data = make_synth(6000)
    dev_data = make_synth(1200)
    test_data = make_synth(1200)

# -----------------------------------------------------------------------------#
# Vocabulary & encoding
all_tokens = {tok for seq in train_data["sequence"] for tok in seq.split()}
vocab = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
vocab.update({"<pad>": 0, "<unk>": 1, "<mask>": 2, "<cls>": 3})
inv_vocab = {i: t for t, i in vocab.items()}
pad_id, unk_id, mask_id, cls_id = (
    vocab[t] for t in ["<pad>", "<unk>", "<mask>", "<cls>"]
)
max_len = max(len(s.split()) for s in train_data["sequence"]) + 1  # +<cls>


def encode(seq: str):
    ids = [cls_id] + [vocab.get(t, unk_id) for t in seq.split()]
    ids = ids[:max_len] + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# -----------------------------------------------------------------------------#
# Datasets
class SeqOnlyDS(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


class LabeledDS(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


# -----------------------------------------------------------------------------#
# Data augmentation (for robustness evaluation only)
def augment(seq: str):
    toks = seq.split()
    if random.random() < 0.5:  # masking
        toks = [t if random.random() > 0.3 else "<mask>" for t in toks]
    else:  # local shuffle
        window = max(1, len(toks) // 4)
        i = random.randint(0, len(toks) - window)
        seg = toks[i : i + window]
        random.shuffle(seg)
        toks[i : i + window] = seg
    return " ".join(toks)


# -----------------------------------------------------------------------------#
# Model
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.gru = nn.GRU(emb, hid, num_layers=n_layers, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h[-1]  # (B,hid)


class Classifier(nn.Module):
    def __init__(self, enc, hid, n_cls):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, x):
        return self.fc(self.enc(x))


# -----------------------------------------------------------------------------#
# DataLoaders
batch_train = 128
train_loader = DataLoader(
    LabeledDS(train_data["sequence"], train_data["label"]),
    batch_size=batch_train,
    shuffle=True,
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
    "no_pretraining": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "val_aca": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "test": {},
        }
    }
}

# -----------------------------------------------------------------------------#
# Training from scratch (No-Pretraining Ablation)
enc = Encoder(len(vocab)).to(device)
model = Classifier(enc, hid=256, n_cls=len(set(train_data["label"]))).to(device)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10


def eval_model(loader):
    model.eval()
    loss_tot = n_tot = 0
    preds, gts, raws = [], [], []
    with torch.no_grad():
        for seqs, labels in loader:
            x = torch.stack([encode(s) for s in seqs]).to(device)
            y = torch.tensor(labels, device=device)
            logits = model(x)
            loss = crit(logits, y)
            loss_tot += loss.item() * y.size(0)
            n_tot += y.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(labels)
            raws.extend(seqs)
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    return loss_tot / n_tot, acc, preds, gts, raws


def aca_metric(seqs, labels, M=3):
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for s, l in zip(seqs, labels):
            variants = [s] + [augment(s) for _ in range(M)]
            xs = torch.stack([encode(v) for v in variants]).to(device)
            p = model(xs).argmax(1).cpu().tolist()
            correct += sum(int(pi == l) for pi in p)
            tot += len(variants)
    return correct / tot


print("\n--- Supervised training (No-Pretraining) ---")
for ep in range(1, epochs + 1):
    # train
    model.train()
    tot_loss = n_items = 0
    correct = 0
    for seqs, labels in train_loader:
        x = torch.stack([encode(s) for s in seqs]).to(device)
        y = torch.tensor(labels, device=device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * y.size(0)
        n_items += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    train_loss = tot_loss / n_items
    train_acc = correct / n_items

    # validation
    val_loss, val_acc, _, _, _ = eval_model(dev_loader)
    val_aca = aca_metric(dev_data["sequence"], dev_data["label"])

    # log
    ed = experiment_data["no_pretraining"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_acc"].append(train_acc)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["val_aca"].append(val_aca)

    print(
        f"Epoch {ep}: train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
        f"ACA={val_aca:.4f}"
    )

# -----------------------------------------------------------------------------#
# Test evaluation
test_loss, test_acc, test_pred, test_gt, test_raw = eval_model(test_loader)
test_swa = shape_weighted_accuracy(test_raw, test_gt, test_pred)
test_cwa = color_weighted_accuracy(test_raw, test_gt, test_pred)
test_aca = aca_metric(test_data["sequence"], test_data["label"])

experiment_data["no_pretraining"]["SPR_BENCH"]["predictions"] = test_pred
experiment_data["no_pretraining"]["SPR_BENCH"]["ground_truth"] = test_gt
experiment_data["no_pretraining"]["SPR_BENCH"]["test"] = dict(
    loss=test_loss, acc=test_acc, swa=test_swa, cwa=test_cwa, aca=test_aca
)

print("\n--- Test results (No-Pretraining) ---")
print(
    f"Loss={test_loss:.4f} | Acc={test_acc:.4f} | SWA={test_swa:.4f} | "
    f"CWA={test_cwa:.4f} | ACA={test_aca:.4f}"
)

# -----------------------------------------------------------------------------#
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
