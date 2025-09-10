# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, csv, math, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# mandatory work dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# experiment tracking skeleton
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------------------------------------------------------------------
# device handling (guideline critical)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# ------------------------- DATA UTILITIES ----------------------------
SHAPES = list("ABCDEF")
COLORS = list("uvwxyz")


def _generate_token():
    return random.choice(SHAPES) + random.choice(COLORS)


def _rule_label(sequence: str) -> str:
    # simple synthetic rule: 'valid' if ≥2 unique shapes else 'invalid'
    return "valid" if len({tok[0] for tok in sequence.split()}) >= 2 else "invalid"


def _write_csv(path: pathlib.Path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence", "label"])
        writer.writerows(rows)


def create_dummy_spr(root: pathlib.Path, n_train=500, n_dev=120, n_test=200):
    print(f"Creating dummy SPR_BENCH at {root.resolve()}")
    rng = random.Random(42)
    root.mkdir(parents=True, exist_ok=True)
    for split, n_rows in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        rows = []
        for idx in range(n_rows):
            seq_len = rng.randint(4, 10)
            seq = " ".join(_generate_token() for _ in range(seq_len))
            rows.append([idx, seq, _rule_label(seq)])
        _write_csv(root / f"{split}.csv", rows)


def ensure_spr_bench() -> pathlib.Path:
    """Locate or create the SPR_BENCH folder with required csv files."""
    # 1) env var
    env_path = os.getenv("SPR_BENCH_PATH")
    candidate_paths = [pathlib.Path(p) for p in ([env_path] if env_path else [])]
    # 2) current dir fallback
    candidate_paths.append(pathlib.Path("./SPR_BENCH"))
    for p in candidate_paths:
        if (
            p
            and (p / "train.csv").exists()
            and (p / "dev.csv").exists()
            and (p / "test.csv").exists()
        ):
            print(f"Found SPR_BENCH at {p.resolve()}")
            return p
    # 3) not found -> create dummy
    dummy_root = pathlib.Path("./SPR_BENCH")
    create_dummy_spr(dummy_root)
    return dummy_root


# --------------------------- helpers ---------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    out["train"] = _load("train.csv")
    out["dev"] = _load("dev.csv")
    out["test"] = _load("test.csv")
    return out


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in sequence.strip().split() if tok})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / max(1e-9, sum(weights))


# --------------------------- dataset ---------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2idx):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        sv = count_shape_variety(seq)
        cv = count_color_variety(seq)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "sym_feats": torch.tensor([sv, cv], dtype=torch.float),
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
    for x in batch:
        pad = max_len - len(x["input_ids"])
        xids = (
            torch.cat([x["input_ids"], torch.zeros(pad, dtype=torch.long)])
            if pad
            else x["input_ids"]
        )
        ids.append(xids)
        labels.append(x["label"])
        feats.append(x["sym_feats"])
        texts.append(x["seq_text"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack(labels),
        "sym_feats": torch.stack(feats),
        "seq_text": texts,
    }


# --------------------------- model -----------------------------------
class HybridSPRModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(512, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.sym_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU())
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, ids, sym_feats):
        B, L = ids.shape
        pos_emb = self.pos[:L].unsqueeze(0).expand(B, L, -1)
        x = self.emb(ids) + pos_emb
        mask = ids == 0
        h = self.encoder(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(1) / (
            (~mask).sum(1, keepdim=True).clamp(min=1)
        )
        sym = self.sym_proj(sym_feats)
        return self.classifier(torch.cat([pooled, sym], dim=-1))


# --------------------------- training utils --------------------------
def train_one_epoch(model, loader, optim, crit):
    model.train()
    total, running = 0, 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = crit(logits, batch["label"])
        loss.backward()
        optim.step()
        running += loss.item() * batch["label"].size(0)
        total += batch["label"].size(0)
    return running / total


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total, running = 0, 0.0
    all_preds, all_labels, all_texts = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = crit(logits, batch["label"])
        running += loss.item() * batch["label"].size(0)
        total += batch["label"].size(0)
        preds = logits.argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().tolist())
        all_texts.extend(batch["seq_text"])
    avg_loss = running / total
    swa = shape_weighted_accuracy(all_texts, all_labels, all_preds)
    return avg_loss, swa, all_texts, all_labels, all_preds


# ------------------------------- run ---------------------------------
def run_experiment():
    # -------- ensure data
    data_root = ensure_spr_bench()
    spr = load_spr_bench(data_root)

    # -------- vocab & datasets
    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    # -------- model / optim
    model = HybridSPRModel(
        len(vocab), d_model=128, nhead=8, num_layers=2, num_classes=len(labels)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # -------- train loop
    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_swa, *_ = evaluate(model, dev_loader, criterion)
        scheduler.step()

        experiment_data["SPR_BENCH"]["epochs"].append(epoch)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)

        print(
            f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f} | SWA = {val_swa:.4f}"
        )

    # -------- test
    test_loss, test_swa, seqs, gts, preds = evaluate(model, test_loader, criterion)
    print(f"\nTEST: loss = {test_loss:.4f} | SWA = {test_swa:.4f}")
    experiment_data["SPR_BENCH"]["metrics"]["test"] = test_swa
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts

    # -------- save
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# trigger
run_experiment()
