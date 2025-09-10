import os, random, csv, pathlib, math, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------------- #
# REQUIRED WORKING DIR + DEVICE HANDLING                                    #
# ------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------- #
# EXPERIMENT TRACKING DICT                                                  #
# ------------------------------------------------------------------------- #
experiment_name = "Remove-Symbolic-Feature-Auxiliary"
experiment_data = {
    experiment_name: {
        "SPR_BENCH": {
            "epochs": [],
            "losses": {"train": [], "val": []},
            "metrics": {
                "SWA": {"train": [], "val": [], "test": None},
                "CWA": {"train": [], "val": [], "test": None},
                "HRG": {"train": [], "val": [], "test": None},
            },
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------------------------------------------------------------- #
# DATASET UTILITIES                                                         #
# ------------------------------------------------------------------------- #
SHAPES, COLORS = list("ABCDEF"), list("uvwxyz")


def _generate_token():
    return random.choice(SHAPES) + random.choice(COLORS)


def _rule_label(sequence: str) -> str:
    # simple rule for dummy data: at least 2 distinct shapes -> valid
    return "valid" if len({tok[0] for tok in sequence.split()}) >= 2 else "invalid"


def _write_csv(path: pathlib.Path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def create_dummy_spr(root: pathlib.Path, n_train=500, n_dev=120, n_test=200):
    print(f"Creating dummy SPR_BENCH at {root}")
    rng = random.Random(42)
    root.mkdir(parents=True, exist_ok=True)
    header = [["id", "sequence", "label"]]
    for split, n_rows in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        rows = []
        for idx in range(n_rows):
            seq_len = rng.randint(4, 10)
            seq = " ".join(_generate_token() for _ in range(seq_len))
            rows.append([idx, seq, _rule_label(seq)])
        _write_csv(root / f"{split}.csv", header + rows)


def ensure_spr_bench() -> pathlib.Path:
    env = os.getenv("SPR_BENCH_PATH")
    for p in filter(None, [env, "./SPR_BENCH"]):
        p = pathlib.Path(p)
        if (
            (p / "train.csv").exists()
            and (p / "dev.csv").exists()
            and (p / "test.csv").exists()
        ):
            print("Found SPR_BENCH at", p.resolve())
            return p
    dummy = pathlib.Path("./SPR_BENCH")
    create_dummy_spr(dummy)
    return dummy


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.strip().split()})


def count_color_variety(seq: str):
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1.0)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1.0)


def harmonic_rule_generalization(swa, cwa):
    return 0.0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)


# ------------------------------------------------------------------------- #
# DATASET & VOCAB                                                           #
# ------------------------------------------------------------------------- #
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
        # placeholder for compatibility; sym_feats unused by ablation model
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": seq,
        }


def build_vocab(sequences):
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in sequences:
        for tok in s.split():
            vocab.setdefault(tok, len(vocab))
    return vocab


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])

    def pad(t):
        if len(t) < max_len:
            return torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)])
        return t

    input_ids = torch.stack([pad(b["input_ids"]) for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    texts = [b["seq_text"] for b in batch]
    return {"input_ids": input_ids, "label": labels, "seq_text": texts}


# ------------------------------------------------------------------------- #
# MODEL                                                                     #
# ------------------------------------------------------------------------- #
class SeqOnlySPRModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(512, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        B, L = ids.shape
        pos_emb = self.pos[:L].unsqueeze(0).expand(B, L, -1)
        x = self.emb(ids) + pos_emb
        mask = ids == 0
        h = self.encoder(x, src_key_padding_mask=mask)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        return self.cls(pooled)


# ------------------------------------------------------------------------- #
# TRAIN & EVAL FUNCTIONS                                                    #
# ------------------------------------------------------------------------- #
def train_epoch(model, loader, opt, crit):
    model.train()
    tot, acc_loss = 0, 0.0
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        loss.backward()
        opt.step()
        acc_loss += loss.item() * batch["label"].size(0)
        tot += batch["label"].size(0)
    return acc_loss / tot


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot, acc_loss = 0, 0.0
    preds, labels, texts = [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        acc_loss += loss.item() * batch["label"].size(0)
        tot += batch["label"].size(0)
        p = logits.argmax(-1).cpu().tolist()
        preds.extend(p)
        labels.extend(batch["label"].cpu().tolist())
        texts.extend(batch["seq_text"])
    swa = shape_weighted_accuracy(texts, labels, preds)
    cwa = color_weighted_accuracy(texts, labels, preds)
    hrg = harmonic_rule_generalization(swa, cwa)
    return acc_loss / tot, swa, cwa, hrg, texts, labels, preds


# ------------------------------------------------------------------------- #
# MAIN EXPERIMENT LOGIC                                                     #
# ------------------------------------------------------------------------- #
def run_experiment():
    root = ensure_spr_bench()
    dset = load_spr_bench(root)
    vocab = build_vocab(dset["train"]["sequence"])
    labels = sorted(set(dset["train"]["label"]))
    lbl2idx = {l: i for i, l in enumerate(labels)}

    train_ds = SPRTorchDataset(dset["train"], vocab, lbl2idx)
    dev_ds = SPRTorchDataset(dset["dev"], vocab, lbl2idx)
    test_ds = SPRTorchDataset(dset["test"], vocab, lbl2idx)

    dl_train = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    dl_dev = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    dl_test = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = SeqOnlySPRModel(
        len(vocab), d_model=128, nhead=8, num_layers=2, num_classes=len(labels)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    num_epochs = 12
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, dl_train, optimizer, criterion)
        val_loss, val_swa, val_cwa, val_hrg, *_ = evaluate(model, dl_dev, criterion)
        scheduler.step()

        ed = experiment_data[experiment_name]["SPR_BENCH"]
        ed["epochs"].append(epoch)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["SWA"]["val"].append(val_swa)
        ed["metrics"]["CWA"]["val"].append(val_cwa)
        ed["metrics"]["HRG"]["val"].append(val_hrg)
        ed["metrics"]["SWA"]["train"].append(None)  # placeholder to align length
        ed["metrics"]["CWA"]["train"].append(None)
        ed["metrics"]["HRG"]["train"].append(None)

        print(
            f"Epoch {epoch:02d}: val_loss={val_loss:.4f} | SWA={val_swa:.4f} | "
            f"CWA={val_cwa:.4f} | HRG={val_hrg:.4f}"
        )

    test_loss, test_swa, test_cwa, test_hrg, _t, gts, preds = evaluate(
        model, dl_test, criterion
    )
    print(
        f"\nTEST: loss={test_loss:.4f} | SWA={test_swa:.4f} | CWA={test_cwa:.4f} | HRG={test_hrg:.4f}"
    )

    ed = experiment_data[experiment_name]["SPR_BENCH"]
    ed["metrics"]["SWA"]["test"] = test_swa
    ed["metrics"]["CWA"]["test"] = test_cwa
    ed["metrics"]["HRG"]["test"] = test_hrg
    ed["predictions"] = preds
    ed["ground_truth"] = gts

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Execute immediately (no __main__ guard per requirements)
run_experiment()
