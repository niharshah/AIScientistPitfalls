import os, pathlib, random, csv, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# experiment tracking dict (ablation-oriented)
experiment_data = {
    "NoPositionalEmbedding": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": None},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
EXP_KEY = ("NoPositionalEmbedding", "SPR_BENCH")  # shorthand

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------- DATA UTILITIES ----------------------------
SHAPES, COLORS = list("ABCDEF"), list("uvwxyz")


def _generate_token():
    return random.choice(SHAPES) + random.choice(COLORS)


def _rule_label(sequence):
    return "valid" if len({tok[0] for tok in sequence.split()}) >= 2 else "invalid"


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence", "label"])
        w.writerows(rows)


def create_dummy_spr(root, n_train=500, n_dev=120, n_test=200):
    print("Creating dummy SPR_BENCH at", root.resolve())
    rng = random.Random(42)
    root.mkdir(parents=True, exist_ok=True)
    for split, n in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        rows = []
        for idx in range(n):
            seq_len = rng.randint(4, 10)
            seq = " ".join(_generate_token() for _ in range(seq_len))
            rows.append([idx, seq, _rule_label(seq)])
        _write_csv(root / f"{split}.csv", rows)


def ensure_spr_bench():
    env = os.getenv("SPR_BENCH_PATH")
    for p in [pathlib.Path(env)] if env else []:
        if p and all((p / f).exists() for f in ["train.csv", "dev.csv", "test.csv"]):
            print("Found SPR_BENCH at", p.resolve())
            return p
    default = pathlib.Path("./SPR_BENCH")
    if not (default / "train.csv").exists():
        create_dummy_spr(default)
    return default


def load_spr_bench(root):
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(w))


# --------------------------- dataset ---------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, l2i):
        self.seqs = split["sequence"]
        self.labels = [l2i[l] for l in split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]
        return {
            "input_ids": torch.tensor(ids),
            "label": torch.tensor(self.labels[idx]),
            "sym_feats": torch.tensor(
                [count_shape_variety(seq), count_color_variety(seq)], dtype=torch.float
            ),
            "seq_text": seq,
        }


def build_vocab(train_sequences):
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in train_sequences:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    ids, labels, feats, texts = [], [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append(
            torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)])
            if pad
            else b["input_ids"]
        )
        labels.append(b["label"])
        feats.append(b["sym_feats"])
        texts.append(b["seq_text"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack(labels),
        "sym_feats": torch.stack(feats),
        "seq_text": texts,
    }


# --------------------------- model -----------------------------------
class HybridSPRModel_NoPos(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, layers, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.sym_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU())
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, ids, sym_feats):
        mask = ids == 0
        h = self.encoder(self.emb(ids), src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        return self.classifier(torch.cat([pooled, self.sym_proj(sym_feats)], -1))


# --------------------------- training utils --------------------------
def train_epoch(model, loader, opt, crit):
    model.train()
    tot, loss_sum = 0, 0.0
    for b in loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in b.items()
        }
        opt.zero_grad()
        loss = crit(model(b["input_ids"], b["sym_feats"]), b["label"].to(device))
        loss.backward()
        opt.step()
        loss_sum += loss.item() * b["label"].size(0)
        tot += b["label"].size(0)
    return loss_sum / tot


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot, loss_sum = 0, 0.0
    preds, labels, texts = [], [], []
    for b in loader:
        b = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in b.items()
        }
        logits = model(b["input_ids"], b["sym_feats"])
        loss = crit(logits, b["label"])
        loss_sum += loss.item() * b["label"].size(0)
        tot += b["label"].size(0)
        p = logits.argmax(-1).cpu().tolist()
        preds += p
        labels += b["label"].cpu().tolist()
        texts += b["seq_text"]
    swa = shape_weighted_accuracy(texts, labels, preds)
    return loss_sum / tot, swa, texts, labels, preds


# ------------------------------- run ---------------------------------
def run_experiment():
    data_root = ensure_spr_bench()
    spr = load_spr_bench(data_root)
    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    l2i = {l: i for i, l in enumerate(labels)}
    train_ds, dev_ds, test_ds = [
        SPRTorchDataset(spr[s], vocab, l2i) for s in ("train", "dev", "test")
    ]
    train_loader = DataLoader(train_ds, 128, True, collate_fn)
    dev_loader = DataLoader(dev_ds, 128, False, collate_fn)
    test_loader = DataLoader(test_ds, 128, False, collate_fn)
    model = HybridSPRModel_NoPos(len(vocab), 128, 8, 2, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    num_epochs = 15
    rec = experiment_data[EXP_KEY[0]][EXP_KEY[1]]
    for ep in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_swa, *_ = evaluate(model, dev_loader, criterion)
        scheduler.step()
        rec["epochs"].append(ep)
        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["train"].append(None)
        rec["metrics"]["val"].append(val_swa)
        print(f"Epoch {ep:02d} | val_loss {val_loss:.4f} | SWA {val_swa:.4f}")
    test_loss, test_swa, seqs, gt, pred = evaluate(model, test_loader, criterion)
    rec["metrics"]["test"] = test_swa
    rec["predictions"] = pred
    rec["ground_truth"] = gt
    print(f"TEST | loss {test_loss:.4f} | SWA {test_swa:.4f}")
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


if __name__ == "__main__":
    run_experiment()
