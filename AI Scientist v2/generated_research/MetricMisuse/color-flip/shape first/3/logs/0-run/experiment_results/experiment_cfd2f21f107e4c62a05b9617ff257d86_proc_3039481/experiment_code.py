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
# Data augmentation (same as baseline)
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
        self.gru = nn.GRU(emb, hid, n_layers, batch_first=True)

    def forward(self, x, seq_output=False):
        z, h = self.gru(self.emb(x))
        return z if seq_output else h[-1]  # (B,L,H) or (B,H)


class Classifier(nn.Module):
    def __init__(self, encoder, hid, n_cls):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, x):
        return self.fc(self.enc(x, seq_output=False))


# -----------------------------------------------------------------------------#
# Helper: create MLM masks
def mask_ids(batch_ids, prob=0.15):
    """
    batch_ids: (B,L) tensor, returns input_ids, labels
    labels == -100 where no prediction
    """
    input_ids = batch_ids.clone()
    labels = batch_ids.clone()
    # mask selection
    mask = (
        (torch.rand(batch_ids.shape, device=batch_ids.device) < prob)
        & (batch_ids != pad_id)
        & (batch_ids != cls_id)
    )
    labels[~mask] = -100
    # 80% -> <mask>
    mask_token_mask = mask & (
        torch.rand(batch_ids.shape, device=batch_ids.device) < 0.8
    )
    input_ids[mask_token_mask] = mask_id
    # 10% -> random
    rand_token_mask = (
        mask
        & (~mask_token_mask)
        & (torch.rand(batch_ids.shape, device=batch_ids.device) < 0.5)
    )
    random_tokens = torch.randint(
        4, len(vocab), size=(rand_token_mask.sum(),), device=device
    )
    input_ids[rand_token_mask] = random_tokens
    # 10% keep unchanged (already in input_ids)
    return input_ids, labels


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
    "MLM_pretrain": {
        "SPR_BENCH": {
            "losses": {"pretrain": [], "train": [], "val": []},
            "metrics": {"val_acc": [], "val_aca": []},
            "test": {},
        }
    }
}

# -----------------------------------------------------------------------------#
# MLM Pre-training
enc = Encoder(len(vocab)).to(device)
mlm_head = nn.Linear(256, len(vocab)).to(device)
opt_enc = torch.optim.Adam(
    list(enc.parameters()) + list(mlm_head.parameters()), lr=1e-3
)
mlm_crit = nn.CrossEntropyLoss(ignore_index=-100)
pre_epochs = 8
print("\n--- MLM Pre-training ---")
for ep in range(1, pre_epochs + 1):
    enc.train()
    mlm_head.train()
    tot = cnt = 0
    t0 = time.time()
    for batch_seqs in pretrain_loader:
        batch_ids = torch.stack([encode(s) for s in batch_seqs]).to(device)
        inp, tgt = mask_ids(batch_ids)
        opt_enc.zero_grad()
        logits = mlm_head(enc(inp, seq_output=True))  # (B,L,V)
        loss = mlm_crit(logits.view(-1, len(vocab)), tgt.view(-1))
        loss.backward()
        opt_enc.step()
        tot += loss.item()
        cnt += 1
    avg = tot / cnt
    experiment_data["MLM_pretrain"]["SPR_BENCH"]["losses"]["pretrain"].append(avg)
    print(f"Epoch {ep}: mlm_loss = {avg:.4f}  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------------#
# Fine-tuning
num_cls = len(set(train_data["label"]))
model = Classifier(enc, hid=256, n_cls=num_cls).to(device)
crit_cls = nn.CrossEntropyLoss()
opt_cls = torch.optim.Adam(model.parameters(), lr=1e-3)
f_epochs = 10


def eval_model(loader):
    model.eval()
    loss_tot = n_tot = 0
    preds, gts, raws = [], [], []
    with torch.no_grad():
        for seqs, labels in loader:
            x = torch.stack([encode(s) for s in seqs]).to(device)
            y = torch.tensor(labels, device=device)
            logits = model(x)
            loss = crit_cls(logits, y)
            loss_tot += loss.item() * y.size(0)
            n_tot += y.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(labels)
            raws.extend(seqs)
    acc = np.mean(np.array(preds) == np.array(gts))
    return loss_tot / n_tot, acc, preds, gts, raws


def aca_metric(seqs, labels, M=3):
    correct = total = 0
    for s, l in zip(seqs, labels):
        variants = [s] + [augment(s) for _ in range(M)]
        xs = torch.stack([encode(v) for v in variants]).to(device)
        with torch.no_grad():
            p = model(xs).argmax(1).cpu().tolist()
        correct += sum(int(pi == l) for pi in p)
        total += len(variants)
    return correct / total


print("\n--- Fine-tuning ---")
for ep in range(1, f_epochs + 1):
    model.train()
    tot = n = 0
    for seqs, labels in train_loader:
        x = torch.stack([encode(s) for s in seqs]).to(device)
        y = torch.tensor(labels, device=device)
        opt_cls.zero_grad()
        loss = crit_cls(model(x), y)
        loss.backward()
        opt_cls.step()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    train_loss = tot / n
    val_loss, val_acc, _, _, _ = eval_model(dev_loader)
    val_aca = aca_metric(dev_data["sequence"], dev_data["label"])
    ed = experiment_data["MLM_pretrain"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["val_acc"].append(val_acc)
    ed["metrics"]["val_aca"].append(val_aca)
    print(
        f"Epoch {ep}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | ACA={val_aca:.4f}"
    )

# -----------------------------------------------------------------------------#
# Test evaluation
test_loss, test_acc, test_pred, test_gt, test_raw = eval_model(test_loader)
test_swa = shape_weighted_accuracy(test_raw, test_gt, test_pred)
test_cwa = color_weighted_accuracy(test_raw, test_gt, test_pred)
test_aca = aca_metric(test_data["sequence"], test_data["label"])
experiment_data["MLM_pretrain"]["SPR_BENCH"]["test"] = {
    "loss": test_loss,
    "acc": test_acc,
    "swa": test_swa,
    "cwa": test_cwa,
    "aca": test_aca,
    "predictions": test_pred,
    "gt": test_gt,
}
print("\n--- Test results ---")
print(
    f"Loss={test_loss:.4f} | Acc={test_acc:.4f} | SWA={test_swa:.4f} | "
    f"CWA={test_cwa:.4f} | ACA={test_aca:.4f}"
)

# -----------------------------------------------------------------------------#
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nExperiment data saved.")
