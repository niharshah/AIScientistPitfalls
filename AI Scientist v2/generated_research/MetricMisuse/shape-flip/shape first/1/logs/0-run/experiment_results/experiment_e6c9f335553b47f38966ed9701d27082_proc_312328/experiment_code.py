import os, pathlib, time, random, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# ----------------------------------------------------------------------
# working dir & reproducibility ----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------------------
# GPU / CPU -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# experiment data skeleton ---------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "val_loss": [],
            "swa": [],
            "cwa": [],
            "hma": [],
            "zs_acc": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ----------------------------------------------------------------------
# data helpers ----------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _load(f"{sp}.csv")
    return out


def build_vocab(dataset) -> Dict[str, int]:
    """token -> id; id 0=<pad>, 1=<unk>"""
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


def count_shape_variety(sequence: str) -> int:
    toks = [t for t in sequence.strip().split() if t not in {"<pad>", "<unk>"}]
    return len(set(tok[0] for tok in toks))


def count_color_variety(sequence: str) -> int:
    toks = [t for t in sequence.strip().split() if t not in {"<pad>", "<unk>"}]
    return len(set(tok[1] for tok in toks if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def harmonic_mean(a, b):
    return 2 * a * b / (a + b) if (a + b) else 0.0


# ----------------------------------------------------------------------
# datasets --------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

vocab = build_vocab(spr["train"])
id2tok = {i: t for t, i in vocab.items()}
print(f"Vocab size: {len(vocab)}")

train_label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_label_set)}
id2label = {i: l for l, i in label2id.items()}
num_train_labels = len(label2id)
print(f"# train labels: {num_train_labels}")


# ----------------------------------------------------------------------
# torch dataset ---------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2id, train_mode=True):
        self.seqs = hf_split["sequence"]
        self.seq_enc = [encode_seq(s, vocab) for s in self.seqs]
        self.labels = hf_split["label"]
        self.train_mode = train_mode
        self.label2id = label2id

    def __len__(self):
        return len(self.seq_enc)

    def __getitem__(self, idx):
        x = torch.tensor(self.seq_enc[idx], dtype=torch.long)
        y_str = self.labels[idx]
        if self.train_mode:
            y = torch.tensor(self.label2id[y_str], dtype=torch.long)
            return {"input": x, "label": y}
        else:
            # unseen labels mapped to -1 sentinel
            y = torch.tensor(self.label2id.get(y_str, -1), dtype=torch.long)
            return {"input": x, "label": y, "label_str": y_str}


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": lens}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    if "label_str" in batch[0]:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], vocab, label2id, True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, False)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id, False)


# ----------------------------------------------------------------------
# model -----------------------------------------------------------------
class SimpleSPRModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, tok_ids, lengths):
        x = self.emb(tok_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # (2,B,H)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.lin(h_cat)


# ----------------------------------------------------------------------
# training helpers ------------------------------------------------------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_ok, total = 0.0, 0, 0
    for batch in loader:
        # move tensor items to device
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss_mask = batch["label"] != -1  # mask for zero-shot samples
        if loss_mask.any():
            loss = criterion(logits[loss_mask], batch["label"][loss_mask])
        else:
            # no supervised instances inside this batch (unlikely)
            loss = torch.zeros(1, device=device, requires_grad=train)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = logits.argmax(1)
            supervised = loss_mask
            if supervised.any():
                total_ok += (
                    (preds[supervised] == batch["label"][supervised]).sum().item()
                )
                total += supervised.sum().item()
            total_loss += loss.item() * loss_mask.sum().item()
    acc = total_ok / total if total else 0.0
    avg_loss = total_loss / total if total else 0.0
    return avg_loss, acc


def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate)
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            raw_input = batch["input"]
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input"], batch["lengths"])
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend([id2label[p] if p in id2label else "UNK" for p in preds])
            all_labels.extend(batch["label_str"])
            # reconstruct sequence
            for seq in raw_input:
                toks = [
                    id2tok.get(tid, "<unk>") for tid in seq.tolist() if tid not in (0,)
                ]
                all_seqs.append(" ".join(toks))
    overall_acc = np.mean([p == t for p, t in zip(all_preds, all_labels)])
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    hma = harmonic_mean(swa, cwa)
    seen = set(train_label_set)
    zs_ids = [i for i, lbl in enumerate(all_labels) if lbl not in seen]
    zs_acc = np.mean([all_preds[i] == all_labels[i] for i in zs_ids]) if zs_ids else 0.0
    return overall_acc, swa, cwa, hma, zs_acc, all_preds, all_labels


# ----------------------------------------------------------------------
# hyper-parameter tuning (batch size only) ------------------------------
BATCH_SIZES = [32, 64, 128, 256, 512]
EPOCHS = 5

for bs in BATCH_SIZES:
    print(f"\n===== Training with batch size {bs} =====")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = SimpleSPRModel(len(vocab), 64, 128, num_train_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optim)
        val_loss, val_acc = run_epoch(model, dev_loader, criterion, None)
        overall_acc, swa, cwa, hma, zs_acc, _, _ = evaluate(model, dev_ds)
        print(
            f"Epoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
            f"val_loss={val_loss:.4f} HMA={hma:.4f}"
        )

        experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["swa"].append(swa)
        experiment_data["SPR_BENCH"]["metrics"]["cwa"].append(cwa)
        experiment_data["SPR_BENCH"]["metrics"]["hma"].append(hma)
        experiment_data["SPR_BENCH"]["metrics"]["zs_acc"].append(zs_acc)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    # final test evaluation ------------------------------------------------
    overall_acc, swa, cwa, hma, zs_acc, preds, truth = evaluate(model, test_ds)
    print(
        f"TEST  acc={overall_acc:.4f} SWA={swa:.4f} CWA={cwa:.4f} "
        f"HMA={hma:.4f} ZSRTA={zs_acc:.4f}"
    )

    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(truth)

# ----------------------------------------------------------------------
# save all experiment data ---------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll experiment data saved to {working_dir}/experiment_data.npy")
