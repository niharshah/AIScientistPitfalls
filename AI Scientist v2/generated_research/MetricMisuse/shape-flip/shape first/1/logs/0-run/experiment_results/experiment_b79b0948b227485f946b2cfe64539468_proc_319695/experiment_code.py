# unidirectional_gru_ablation.py
# -------------------------------------------------------------
# UniDirectional-GRU ablation for SPR-BENCH
# -------------------------------------------------------------
import os, pathlib, time, random, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# -------------------------------------------------------------
# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------
# experiment container
experiment_data = {"UniDirectional_GRU": {"SPR_BENCH": {}}}

# -------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------
# util helpers
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname: str):
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    out["train"] = _load("train.csv")
    out["dev"] = _load("dev.csv")
    out["test"] = _load("test.csv")
    return out


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------------------------------------------------
# load dataset (adjust path if needed)
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# -------------------------------------------------------------
# vocab & label mapping
def build_vocab(ds) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in ds["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")

train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"# seen rule labels: {num_labels}")


def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


# -------------------------------------------------------------
# torch dataset / collate
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
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": lens}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], vocab, label2id, True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, True)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id, False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# -------------------------------------------------------------
# model (UniDirectional GRU)
class SimpleSPRModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_labels: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # Uni-directional GRU
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=False)
        # Linear takes only hidden_dim (not doubled)
        self.lin = nn.Linear(hidden_dim, num_labels)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h shape: (1, B, H)
        h = h.squeeze(0)  # (B, H)
        return self.lin(h)


# -------------------------------------------------------------
# training helpers
def run_epoch(model, loader, criterion, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    tot_loss = tot_ok = tot = 0
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            lbl = batch["label"].to(device)
            logits = model(inp, lens)
            loss = criterion(logits, lbl)
            if train_mode:
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
    preds, gold, seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(inp, lens)
            p = logits.argmax(1).cpu().tolist()
            preds.extend([id2label.get(i, "UNK") for i in p])
            gold.extend(batch["label_str"])
            seqs.extend(
                [
                    " ".join(
                        [list(vocab.keys())[tok] for tok in s.tolist() if tok != 0]
                    )
                    for s in batch["input"]
                ]
            )
    overall = np.mean([p == g for p, g in zip(preds, gold)])
    swa = shape_weighted_accuracy(seqs, gold, preds)
    cwa = color_weighted_accuracy(seqs, gold, preds)
    seen = set(train_labels)
    zs_idx = [i for i, g in enumerate(gold) if g not in seen]
    zs_acc = np.mean([preds[i] == gold[i] for i in zs_idx]) if zs_idx else float("nan")
    return overall, swa, cwa, zs_acc, preds, gold


# -------------------------------------------------------------
# hyper-parameter tuning
hidden_dims = [64, 128, 256, 512]
EPOCHS = 5
for hd in hidden_dims:
    print(f"\n=== UniDir GRU | hidden_dim={hd} ===")
    model = SimpleSPRModel(len(vocab), 64, hd, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
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
    # final test
    overall, swa, cwa, zs_acc, all_preds, all_gold = evaluate_test(model)
    mdata["metrics"]["ZSRTA"].append(zs_acc)
    mdata["predictions"] = all_preds
    mdata["ground_truth"] = all_gold
    print(
        f"TEST Acc={overall:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | ZSRTA={zs_acc:.4f}"
    )
    experiment_data["UniDirectional_GRU"]["SPR_BENCH"][f"hidden_{hd}"] = mdata

# -------------------------------------------------------------
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
