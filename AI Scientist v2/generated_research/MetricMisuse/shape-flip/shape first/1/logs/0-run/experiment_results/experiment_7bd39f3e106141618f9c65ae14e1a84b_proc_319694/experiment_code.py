# no_length_masking_ablation.py
import os, pathlib, time, random, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# ----------------------------------------------------------------------
# reproducibility -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------
# experiment container --------------------------------------------------
experiment_data = {"no_length_masking": {"SPR_BENCH": {}}}
# ----------------------------------------------------------------------
# working dir / save path ----------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
# ----------------------------------------------------------------------
# device ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------
# util functions --------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# ----------------------------------------------------------------------
# load data -------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ----------------------------------------------------------------------
# vocab + label mapping -------------------------------------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")


def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"# seen rule labels: {num_labels}")


# ----------------------------------------------------------------------
# Torch dataset ---------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id, train_mode=True):
        self.seq_enc = [encode_seq(s, vocab) for s in split["sequence"]]
        self.labels = split["label"]
        self.train_mode = train_mode
        self.label2id = label2id

    def __len__(self):
        return len(self.seq_enc)

    def __getitem__(self, idx):
        x = torch.tensor(self.seq_enc[idx], dtype=torch.long)
        if self.train_mode:
            y = torch.tensor(self.label2id[self.labels[idx]], dtype=torch.long)
            return {"input": x, "label": y}
        else:
            return {"input": x, "label_str": self.labels[idx]}


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = [len(x) for x in xs]
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": torch.tensor(lens, dtype=torch.long)}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


# loaders (reuse across experiments)
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, label2id, True),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"], vocab, label2id, True),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2id, False),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)


# ----------------------------------------------------------------------
# Model without length masking -----------------------------------------
class SimpleSPRModelNoMask(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x, lengths=None):  # lengths kept for signature compatibility
        e = self.emb(x)  # [B,T,E]
        _, h = self.gru(e)  # no pack_padded_sequence
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # [B, 2H]
        return self.lin(h_cat)


# ----------------------------------------------------------------------
# training / eval helpers ----------------------------------------------
def run_epoch(model, loader, criterion, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    tot_loss = tot_ok = tot = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)  # not used but passed
            lbl = batch["label"].to(device)
            logits = model(inp, lens)
            loss = criterion(logits, lbl)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * inp.size(0)
            preds = logits.argmax(1)
            tot_ok += (preds == lbl).sum().item()
            tot += inp.size(0)
    return tot_loss / tot, tot_ok / tot


def evaluate_test(model):
    model.eval()
    preds_all, labels_all, seqs_all = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(inp, lens)
            preds = logits.argmax(1).cpu().tolist()
            labels_all.extend(batch["label_str"])
            preds_all.extend([id2label.get(p, "UNK") for p in preds])
            for seq in batch["input"]:
                seqs_all.append(
                    " ".join(
                        [list(vocab.keys())[tok] for tok in seq.tolist() if tok != 0]
                    )
                )
    oa = np.mean([p == t for p, t in zip(preds_all, labels_all)])
    swa = shape_weighted_accuracy(seqs_all, labels_all, preds_all)
    cwa = color_weighted_accuracy(seqs_all, labels_all, preds_all)
    zs_idx = [i for i, l in enumerate(labels_all) if l not in train_labels]
    zs_acc = (
        np.mean([preds_all[i] == labels_all[i] for i in zs_idx])
        if zs_idx
        else float("nan")
    )
    return oa, swa, cwa, zs_acc, preds_all, labels_all


# ----------------------------------------------------------------------
# hyperparameter sweep --------------------------------------------------
hidden_dims = [64, 128, 256, 512]
EPOCHS = 5
for hd in hidden_dims:
    print(f"\n=== No-masking run: hidden_dim={hd} ===")
    model = SimpleSPRModelNoMask(len(vocab), 64, hd, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mdata = {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "ZSRTA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt)
        val_loss, val_acc = run_epoch(model, dev_loader, criterion)
        mdata["metrics"]["train_acc"].append(tr_acc)
        mdata["metrics"]["val_acc"].append(val_acc)
        mdata["metrics"]["val_loss"].append(val_loss)
        mdata["losses"]["train"].append(tr_loss)
        mdata["losses"]["val"].append(val_loss)
        mdata["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )

    overall_acc, swa, cwa, zs_acc, preds_all, labels_all = evaluate_test(model)
    mdata["metrics"]["ZSRTA"].append(zs_acc)
    mdata["predictions"] = preds_all
    mdata["ground_truth"] = labels_all
    print(
        f"TEST Acc={overall_acc:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | ZSRTA={zs_acc:.4f}"
    )
    experiment_data["no_length_masking"]["SPR_BENCH"][f"hidden_{hd}"] = mdata

# ----------------------------------------------------------------------
# save ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {working_dir}/experiment_data.npy")
