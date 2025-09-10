import os, random, pathlib, math, time, copy, numpy as np, torch
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
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


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
    train_data, dev_data, test_data = (
        make_synth(6000),
        make_synth(1200),
        make_synth(1200),
    )

# -----------------------------------------------------------------------------#
# Vocabulary
all_tokens = {tok for seq in train_data["sequence"] for tok in seq.split()}
vocab = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
vocab.update({"<pad>": 0, "<unk>": 1, "<mask>": 2, "<cls>": 3})
inv_vocab = {i: t for t, i in vocab.items()}
pad_id, unk_id, mask_id, cls_id = [
    vocab[t] for t in ["<pad>", "<unk>", "<mask>", "<cls>"]
]
max_len = max(len(s.split()) for s in train_data["sequence"]) + 1


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
# Data augmentation
def augment(seq: str):
    toks = seq.split()
    if random.random() < 0.5:
        toks = [t if random.random() > 0.3 else "<mask>" for t in toks]
    else:
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
        z, h = self.gru(self.emb(x))
        return h[-1]


class Classifier(nn.Module):
    def __init__(self, encoder, hid, n_cls):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, x):
        return self.fc(self.enc(x))


# -----------------------------------------------------------------------------#
# Contrastive loss
def contrastive_loss(z, temp=0.1):
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temp
    N = sim.size(0)
    sim.masked_fill_(torch.eye(N, device=z.device).bool(), -9e15)
    pos_idx = torch.arange(N, device=z.device) ^ 1
    return nn.functional.cross_entropy(sim, pos_idx)


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
    "full_tune": {
        "SPR_BENCH": {
            "losses": {"pretrain": [], "train": [], "val": []},
            "metrics": {"val_acc": [], "val_aca": []},
            "test": {},
        }
    },
    "frozen_encoder": {
        "SPR_BENCH": {
            "losses": {"train": [], "val": []},
            "metrics": {"val_acc": [], "val_aca": []},
            "test": {},
        }
    },
}

# -----------------------------------------------------------------------------#
# Pre-training
enc = Encoder(len(vocab)).to(device)
opt_enc = torch.optim.Adam(enc.parameters(), lr=1e-3)
pre_epochs = 8
print("\n--- Contrastive Pre-training ---")
for ep in range(1, pre_epochs + 1):
    enc.train()
    tot = cnt = 0
    t0 = time.time()
    for batch_seqs in pretrain_loader:
        views = []
        for s in batch_seqs:
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
    experiment_data["full_tune"]["SPR_BENCH"]["losses"]["pretrain"].append(avg)
    print(f"Epoch {ep}: contrastive_loss={avg:.4f} ({time.time()-t0:.1f}s)")

# Save a snapshot of the encoder just after pre-training
enc_pretrained_state = copy.deepcopy(enc.state_dict())

# -----------------------------------------------------------------------------#
# Helper functions
crit = nn.CrossEntropyLoss()


def eval_model(model, loader):
    model.eval()
    loss_tot = n_tot = 0
    preds = []
    gts = []
    raws = []
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
    acc = np.mean(np.array(preds) == np.array(gts))
    return loss_tot / n_tot, acc, preds, gts, raws


def aca_metric(model, seqs, labels, M=3):
    correct = total = 0
    for s, l in zip(seqs, labels):
        variants = [s] + [augment(s) for _ in range(M)]
        xs = torch.stack([encode(v) for v in variants]).to(device)
        with torch.no_grad():
            p = model(xs).argmax(1).cpu().tolist()
        correct += sum(int(pi == l) for pi in p)
        total += len(variants)
    return correct / total


# -----------------------------------------------------------------------------#
# ---------------- Full fine-tuning (baseline) --------------------------------#
num_cls = len(set(train_data["label"]))
enc_full = Encoder(len(vocab)).to(device)
enc_full.load_state_dict(enc_pretrained_state)
model_full = Classifier(enc_full, hid=256, n_cls=num_cls).to(device)
opt_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
f_epochs = 10
print("\n--- Fine-tuning (full_tune) ---")
for ep in range(1, f_epochs + 1):
    model_full.train()
    tot = n = 0
    for seqs, labels in train_loader:
        x = torch.stack([encode(s) for s in seqs]).to(device)
        y = torch.tensor(labels, device=device)
        opt_full.zero_grad()
        loss = crit(model_full(x), y)
        loss.backward()
        opt_full.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = tot / n
    val_loss, val_acc, _, _, _ = eval_model(model_full, dev_loader)
    val_aca = aca_metric(model_full, dev_data["sequence"], dev_data["label"])
    experiment_data["full_tune"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["full_tune"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["full_tune"]["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["full_tune"]["SPR_BENCH"]["metrics"]["val_aca"].append(val_aca)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | ACA={val_aca:.4f}"
    )

# Test baseline
tl, ta, tp, tg, tr = eval_model(model_full, test_loader)
swa = shape_weighted_accuracy(tr, tg, tp)
cwa = color_weighted_accuracy(tr, tg, tp)
aca = aca_metric(model_full, test_data["sequence"], test_data["label"])
experiment_data["full_tune"]["SPR_BENCH"]["test"] = {
    "loss": tl,
    "acc": ta,
    "swa": swa,
    "cwa": cwa,
    "aca": aca,
    "predictions": tp,
    "ground_truth": tg,
}
print("\n--- Test results (full_tune) ---")
print(f"Loss={tl:.4f} | Acc={ta:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | ACA={aca:.4f}")

# -----------------------------------------------------------------------------#
# --------------- Frozen-encoder fine-tuning (ablation) -----------------------#
enc_frozen = Encoder(len(vocab)).to(device)
enc_frozen.load_state_dict(enc_pretrained_state)
for p in enc_frozen.parameters():
    p.requires_grad = False  # freeze
model_frozen = Classifier(enc_frozen, hid=256, n_cls=num_cls).to(device)
opt_frozen = torch.optim.Adam(
    model_frozen.fc.parameters(), lr=1e-3
)  # only classifier params
print("\n--- Fine-tuning (frozen_encoder) ---")
for ep in range(1, f_epochs + 1):
    model_frozen.train()
    tot = n = 0
    for seqs, labels in train_loader:
        x = torch.stack([encode(s) for s in seqs]).to(device)
        y = torch.tensor(labels, device=device)
        opt_frozen.zero_grad()
        loss = crit(model_frozen(x), y)
        loss.backward()
        opt_frozen.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = tot / n
    val_loss, val_acc, _, _, _ = eval_model(model_frozen, dev_loader)
    val_aca = aca_metric(model_frozen, dev_data["sequence"], dev_data["label"])
    experiment_data["frozen_encoder"]["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["frozen_encoder"]["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["frozen_encoder"]["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["frozen_encoder"]["SPR_BENCH"]["metrics"]["val_aca"].append(val_aca)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | ACA={val_aca:.4f}"
    )

# Test ablation
tl, ta, tp, tg, tr = eval_model(model_frozen, test_loader)
swa = shape_weighted_accuracy(tr, tg, tp)
cwa = color_weighted_accuracy(tr, tg, tp)
aca = aca_metric(model_frozen, test_data["sequence"], test_data["label"])
experiment_data["frozen_encoder"]["SPR_BENCH"]["test"] = {
    "loss": tl,
    "acc": ta,
    "swa": swa,
    "cwa": cwa,
    "aca": aca,
    "predictions": tp,
    "ground_truth": tg,
}
print("\n--- Test results (frozen_encoder) ---")
print(f"Loss={tl:.4f} | Acc={ta:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | ACA={aca:.4f}")

# -----------------------------------------------------------------------------#
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "\nAll experiment data saved to", os.path.join(working_dir, "experiment_data.npy")
)
