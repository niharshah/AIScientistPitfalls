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

import os, pathlib, math, time, warnings, random, subprocess, sys
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------- #
# mandatory working directory & device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ---------------------------------------------------------------------- #
# optional dependency handling
try:
    from datasets import load_dataset, DatasetDict
except ImportError:
    print("`datasets` library missing – installing…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset, DatasetDict
# ---------------------------------------------------------------------- #
# experiment bookkeeping container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}
save_slot = experiment_data["SPR_BENCH"]
# ---------------------------------------------------------------------- #
warnings.filterwarnings("ignore")  # keep logs tidy


def _create_synthetic_spr_bench(dst: pathlib.Path, n_train=2000, n_dev=400, n_test=800):
    """Generate a toy SPR_BENCH folder if the real one is absent."""
    print(f"Creating synthetic SPR_BENCH at {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    def _make_split(n_rows, fname):
        toks = list("abcdefghij")
        rng = random.Random(42)
        with open(dst / fname, "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n_rows):
                seq_len = rng.randint(4, 12)
                seq = " ".join(rng.choices(toks, k=seq_len))
                # arbitrary rule: label is parity of 'a' count mod 4
                label = seq.split().count("a") % 4
                f.write(f"{i},{seq},{label}\n")

    _make_split(n_train, "train.csv")
    _make_split(n_dev, "dev.csv")
    _make_split(n_test, "test.csv")


def _locate_or_build_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
    ]
    for c in candidates:
        if (
            c
            and c.exists()
            and {"train.csv", "dev.csv", "test.csv"}.issubset(
                {p.name for p in c.iterdir()}
            )
        ):
            print(f"Found SPR_BENCH at {c}")
            return c
    # not found – create synthetic in working dir
    synthetic_root = pathlib.Path(working_dir) / "SPR_BENCH"
    _create_synthetic_spr_bench(synthetic_root)
    return synthetic_root


root = _locate_or_build_spr_bench()


def load_spr_bench(path: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(path / csv_name),
            split="train",
            cache_dir=os.path.join(working_dir, ".cache_dsets"),
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(root)


# ---------------------------------------------------------------------- #
# vocabulary & dataset wrappers
class SPRTokenDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.ds, self.vocab = hf_ds, vocab
        self.pad, self.cls = vocab["<pad>"], vocab["<cls>"]

    def __len__(self):
        return len(self.ds)

    def _encode(self, seq):
        return [self.cls] + [self.vocab[t] for t in seq.strip().split()]

    def __getitem__(self, idx):
        row = self.ds[idx]
        ids = torch.tensor(self._encode(row["sequence"]), dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {"input_ids": ids, "labels": label}


def build_vocab(train_split):
    vocab = {"<pad>": 0, "<cls>": 1}
    for ex in train_split:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocabulary size:", len(vocab))


def collate_fn(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


train_ds, dev_ds, test_ds = (
    SPRTokenDataset(spr[s], vocab) for s in ("train", "dev", "test")
)
train_loader = DataLoader(
    train_ds, 128, shuffle=True, collate_fn=lambda b: collate_fn(b, vocab["<pad>"])
)
dev_loader = DataLoader(
    dev_ds, 256, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab["<pad>"])
)
test_loader = DataLoader(
    test_ds, 256, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab["<pad>"])
)

num_labels = len({int(ex["label"]) for ex in spr["train"]})
max_len = max(len(ex["sequence"].split()) + 1 for ex in spr["train"])
print("Max token length:", max_len)


# ---------------------------------------------------------------------- #
# model definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (B,L,D)
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=4,
        ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.pos(self.embed(input_ids))
        x = self.enc(x, src_key_padding_mask=(attention_mask == 0))
        cls = self.norm(x[:, 0])
        return self.cls_head(cls)


model = SPRTransformer(len(vocab), num_labels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ---------------------------------------------------------------------- #
def macro_f1(preds: torch.Tensor, labels: torch.Tensor, n_cls: int):
    p, l = preds.cpu().numpy(), labels.cpu().numpy()
    f1s = []
    for c in range(n_cls):
        tp = ((p == c) & (l == c)).sum()
        fp = ((p == c) & (l != c)).sum()
        fn = ((p != c) & (l == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    tot_loss = tot_correct = tot_cnt = 0
    preds_all, labels_all = [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        preds = logits.argmax(1).detach()
        preds_all.append(preds)
        labels_all.append(batch["labels"])
        tot_loss += loss.item() * preds.size(0)
        tot_correct += (preds == batch["labels"]).sum().item()
        tot_cnt += preds.size(0)
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    acc = tot_correct / tot_cnt
    f1 = macro_f1(preds_all, labels_all, num_labels)
    return tot_loss / tot_cnt, acc, f1


# ---------------------------------------------------------------------- #
best_f1, patience, wait = 0, 2, 0
epochs = 10
for ep in range(1, epochs + 1):
    t_loss, t_acc, t_f1 = run_epoch(train_loader, True)
    v_loss, v_acc, v_f1 = run_epoch(dev_loader, False)
    # store metrics
    save_slot["losses"]["train"].append(t_loss)
    save_slot["losses"]["val"].append(v_loss)
    save_slot["metrics"]["train_acc"].append(t_acc)
    save_slot["metrics"]["val_acc"].append(v_acc)
    save_slot["metrics"]["train_f1"].append(t_f1)
    save_slot["metrics"]["val_f1"].append(v_f1)
    print(
        f"Epoch {ep}: val_loss = {v_loss:.4f} | val_acc = {v_acc*100:.2f}% | val_F1 = {v_f1:.4f}"
    )
    if v_f1 > best_f1:
        best_f1, wait = v_f1, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# ---------------------------------------------------------------------- #
# Test evaluation with best model
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.eval()
test_preds, test_lbls = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_preds.append(logits.argmax(1).cpu())
        test_lbls.append(batch["labels"].cpu())
test_preds = torch.cat(test_preds)
test_lbls = torch.cat(test_lbls)
test_acc = (test_preds == test_lbls).float().mean().item()
test_f1 = macro_f1(test_preds, test_lbls, num_labels)
print(f"Test accuracy: {test_acc*100:.2f}% | Test macroF1: {test_f1:.4f}")

save_slot["predictions"] = test_preds.numpy()
save_slot["ground_truth"] = test_lbls.numpy()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all experiment data to working/experiment_data.npy")
