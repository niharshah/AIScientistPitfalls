# ============================ SET-UP ========================================
import os, pathlib, random, csv, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -------- working dir -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- GLOBAL DEVICE PRINT --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================== EXPERIMENT DATA =================================
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train": {"SWA": [], "CWA": [], "HRG": []},
            "val": {"SWA": [], "CWA": [], "HRG": []},
            "test": {"SWA": None, "CWA": None, "HRG": None},
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "timestamp": time.asctime(),
    }
}
# ============================================================================

# ------------------------- DATA UTILITIES -----------------------------------
SHAPES = list("ABCDEF")
COLORS = list("uvwxyz")
rng_global = random.Random(42)


def _generate_token():
    return rng_global.choice(SHAPES) + rng_global.choice(COLORS)


def _rule_label(sequence: str) -> str:
    # very simple dummy rule: at least 2 distinct shapes -> valid
    return "valid" if len({tok[0] for tok in sequence.split()}) >= 2 else "invalid"


def _write_csv(path: pathlib.Path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def create_dummy_spr(root: pathlib.Path, n_train=500, n_dev=120, n_test=200):
    print(f"Creating dummy SPR_BENCH at {root}")
    root.mkdir(parents=True, exist_ok=True)
    header = [["id", "sequence", "label"]]
    for split, n_rows in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        rows = []
        for idx in range(n_rows):
            seq_len = rng_global.randint(4, 10)
            seq = " ".join(_generate_token() for _ in range(seq_len))
            rows.append([idx, seq, _rule_label(seq)])
        _write_csv(root / f"{split}.csv", header + rows)


def ensure_spr_bench() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_PATH")
    for p in filter(None, [env, "./SPR_BENCH"]):
        p = pathlib.Path(p)
        if all((p / f"{s}.csv").exists() for s in ["train", "dev", "test"]):
            print("Found SPR_BENCH at", p.resolve())
            return p
    dummy = pathlib.Path("./SPR_BENCH")
    create_dummy_spr(dummy)
    return dummy


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split_name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split()})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred) -> float:
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) + 1e-9)


def color_weighted_accuracy(seqs, y_true, y_pred) -> float:
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) + 1e-9)


def harmonic_rule_generalization(swa: float, cwa: float) -> float:
    if swa == 0 or cwa == 0:
        return 0.0
    return 2 * swa * cwa / (swa + cwa)


# ------------------------- DATASET & VOCAB -----------------------------------
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
        sv, cv = count_shape_variety(seq), count_color_variety(seq)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": seq,
            "sym_feats": torch.tensor([sv, cv], dtype=torch.float),
        }


def build_vocab(train_sequences):
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in train_sequences:
        for tok in s.split():
            vocab.setdefault(tok, len(vocab))
    return vocab


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])

    def pad(x):
        if len(x) < max_len:
            return torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
        return x

    ids = torch.stack([pad(b["input_ids"]) for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    texts = [b["seq_text"] for b in batch]
    feats = torch.stack([b["sym_feats"] for b in batch])
    return {"input_ids": ids, "label": labels, "seq_text": texts, "sym_feats": feats}


# ----------------------------- MODEL -----------------------------------------
class SeqOnlySPRModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(512, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        B, L = ids.shape
        x = self.emb(ids) + self.pos[:L].unsqueeze(0)
        mask = ids == 0
        h = self.encoder(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(1) / (
            (~mask).sum(1, keepdim=True).clamp(min=1)
        )
        return self.classifier(pooled)


# ----------------------- TRAIN / EVAL UTILS ----------------------------------
def move_batch_to_device(batch):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def train_one_epoch(model, loader, optim, crit):
    model.train()
    tot_items, acc_loss = 0, 0.0
    for batch in loader:
        batch = move_batch_to_device(batch)
        optim.zero_grad()
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        loss.backward()
        optim.step()
        acc_loss += loss.item() * batch["label"].size(0)
        tot_items += batch["label"].size(0)
    return acc_loss / tot_items


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot_items, acc_loss = 0, 0.0
    preds, labels, texts = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch)
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        acc_loss += loss.item() * batch["label"].size(0)
        tot_items += batch["label"].size(0)
        pr = logits.argmax(-1).cpu().tolist()
        preds.extend(pr)
        labels.extend(batch["label"].cpu().tolist())
        texts.extend(batch["seq_text"])
    swa = shape_weighted_accuracy(texts, labels, preds)
    cwa = color_weighted_accuracy(texts, labels, preds)
    hrg = harmonic_rule_generalization(swa, cwa)
    return acc_loss / tot_items, swa, cwa, hrg, texts, labels, preds


# ------------------------------ MAIN RUN -------------------------------------
def run_experiment():
    root = ensure_spr_bench()
    dset = load_spr_bench(root)

    vocab = build_vocab(dset["train"]["sequence"])
    labels_sorted = sorted(set(dset["train"]["label"]))
    lbl2idx = {l: i for i, l in enumerate(labels_sorted)}

    train_ds = SPRTorchDataset(dset["train"], vocab, lbl2idx)
    dev_ds = SPRTorchDataset(dset["dev"], vocab, lbl2idx)
    test_ds = SPRTorchDataset(dset["test"], vocab, lbl2idx)

    dl_train = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    dl_dev = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    dl_test = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = SeqOnlySPRModel(
        vocab_size=len(vocab),
        d_model=128,
        nhead=8,
        num_layers=2,
        num_classes=len(labels_sorted),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        t_loss = train_one_epoch(model, dl_train, optimizer, criterion)
        v_loss, v_swa, v_cwa, v_hrg, *_ = evaluate(model, dl_dev, criterion)
        scheduler.step()

        ed = experiment_data["SPR_BENCH"]
        ed["epochs"].append(epoch)
        ed["losses"]["train"].append(t_loss)
        ed["losses"]["val"].append(v_loss)
        ed["metrics"]["train"]["SWA"].append(None)
        ed["metrics"]["train"]["CWA"].append(None)
        ed["metrics"]["train"]["HRG"].append(None)
        ed["metrics"]["val"]["SWA"].append(v_swa)
        ed["metrics"]["val"]["CWA"].append(v_cwa)
        ed["metrics"]["val"]["HRG"].append(v_hrg)

        print(
            f"Epoch {epoch:02d}: val_loss={v_loss:.4f} | "
            f"SWA={v_swa:.4f} | CWA={v_cwa:.4f} | HRG={v_hrg:.4f}"
        )

    # ----------- TEST EVALUATION --------------------------------------------
    test_loss, t_swa, t_cwa, t_hrg, _, gts, preds = evaluate(model, dl_test, criterion)
    print(
        f"\nTEST: loss={test_loss:.4f} | SWA={t_swa:.4f} | CWA={t_cwa:.4f} | HRG={t_hrg:.4f}"
    )

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["test"]["SWA"] = t_swa
    ed["metrics"]["test"]["CWA"] = t_cwa
    ed["metrics"]["test"]["HRG"] = t_hrg
    ed["predictions"] = preds
    ed["ground_truth"] = gts

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(f"Saved experiment data to {working_dir}/experiment_data.npy")


# ------------------------------- EXECUTE -------------------------------------
run_experiment()
