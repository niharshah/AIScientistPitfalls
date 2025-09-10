# multi_synth_ablation.py
import os, pathlib, time, random, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import (
    DatasetDict,
    load_dataset,
    Dataset as HFDataset,
    concatenate_datasets,
)
from typing import List, Dict

# -------------------- reproducibility ---------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- experiment container ----------------------------
experiment_data = {"multi_dataset_training": {"SPR_BENCH": {}, "NOISE": {}, "LONG": {}}}

# -------------------- working dir -------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- helpers for accuracies --------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# -------------------- load canonical SPR_BENCH ------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("SPR_BENCH sizes:", {k: len(v) for k, v in spr.items()})


# -------------------- create synthetic variants -----------------------
def create_noise_variant(ref: DatasetDict, token_pool: List[str], p=0.1) -> DatasetDict:
    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        seqs, labels = [], []
        for s, lbl in zip(ref[split]["sequence"], ref[split]["label"]):
            toks = s.strip().split()
            toks = [
                t if random.random() > p else random.choice(token_pool) for t in toks
            ]
            seqs.append(" ".join(toks))
            labels.append(lbl)
        out[split] = HFDataset.from_dict({"sequence": seqs, "label": labels})
    return out


def create_long_variant(ref: DatasetDict, min_rep=2, max_rep=4) -> DatasetDict:
    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        seqs, labels = [], []
        for s, lbl in zip(ref[split]["sequence"], ref[split]["label"]):
            toks = s.strip().split()
            rep = random.randint(min_rep, max_rep)
            seqs.append(" ".join(toks * rep))
            labels.append(lbl)
        out[split] = HFDataset.from_dict({"sequence": seqs, "label": labels})
    return out


token_pool = list(
    {tok for seq in spr["train"]["sequence"] for tok in seq.strip().split()}
)
noise_ds = create_noise_variant(spr, token_pool, p=0.1)
long_ds = create_long_variant(spr, 2, 4)
print("NOISE sizes:", {k: len(v) for k, v in noise_ds.items()})
print("LONG sizes :", {k: len(v) for k, v in long_ds.items()})

# -------------------- joint train/val splits --------------------------
combo = DatasetDict()
combo["train"] = concatenate_datasets(
    [spr["train"], noise_ds["train"], long_ds["train"]]
)
combo["dev"] = concatenate_datasets([spr["dev"], noise_ds["dev"], long_ds["dev"]])


# -------------------- vocab / label mapping on combined ---------------
def build_vocab(dataset) -> Dict[str, int]:
    v = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in v:
                v[tok] = len(v)
    return v


vocab = build_vocab(combo["train"])
id2tok = {i: t for t, i in vocab.items()}
print(f"Combined vocab size: {len(vocab)}")

train_labels = sorted(set(combo["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"# rule labels: {num_labels}")


# -------------------- Torch dataset / dataloaders ---------------------
def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


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
            return {
                "input": x,
                "label": torch.tensor(
                    self.label2id[self.labels[idx]], dtype=torch.long
                ),
            }
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


train_loader = DataLoader(
    SPRTorchDataset(combo["train"], vocab, label2id, True),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(combo["dev"], vocab, label2id, True),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)

# test loaders per corpus
test_loaders = {
    "SPR_BENCH": DataLoader(
        SPRTorchDataset(spr["test"], vocab, label2id, False),
        batch_size=256,
        shuffle=False,
        collate_fn=collate,
    ),
    "NOISE": DataLoader(
        SPRTorchDataset(noise_ds["test"], vocab, label2id, False),
        batch_size=256,
        shuffle=False,
        collate_fn=collate,
    ),
    "LONG": DataLoader(
        SPRTorchDataset(long_ds["test"], vocab, label2id, False),
        batch_size=256,
        shuffle=False,
        collate_fn=collate,
    ),
}


# -------------------- Model -------------------------------------------
class SimpleSPRModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        return self.lin(h_cat)


# -------------------- train / eval helpers ----------------------------
def run_epoch(model, loader, criterion, opt=None):
    training = opt is not None
    model.train() if training else model.eval()
    tot_loss = tot_ok = tot = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            lbl = batch["label"].to(device)
            logits = model(inp, lens)
            loss = criterion(logits, lbl)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * inp.size(0)
            tot_ok += (logits.argmax(1) == lbl).sum().item()
            tot += inp.size(0)
    return tot_loss / tot, tot_ok / tot


def evaluate_test(model, loader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(inp, lens)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend([id2label[p] for p in preds])
            all_labels.extend(batch["label_str"])
            for seq in batch["input"]:
                toks = [id2tok[tok] for tok in seq.tolist() if tok != 0]
                all_seqs.append(" ".join(toks))
    acc = np.mean([p == t for p, t in zip(all_preds, all_labels)])
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    seen_rules = set(train_labels)
    zs_idx = [i for i, lbl in enumerate(all_labels) if lbl not in seen_rules]
    zs_acc = (
        np.mean([all_preds[i] == all_labels[i] for i in zs_idx])
        if zs_idx
        else float("nan")
    )
    return acc, swa, cwa, zs_acc, all_preds, all_labels


# -------------------- hyper-parameter loop ----------------------------
hidden_dims = [64, 128, 256, 512]
EPOCHS = 5
for hd in hidden_dims:
    print(f"\n=== Hidden dim {hd} ===")
    model = SimpleSPRModel(len(vocab), 64, hd, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    history = {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
    }
    # training
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, crit, opt)
        val_loss, val_acc = run_epoch(model, dev_loader, crit)
        history["metrics"]["train_acc"].append(tr_acc)
        history["metrics"]["val_acc"].append(val_acc)
        history["losses"]["train"].append(tr_loss)
        history["losses"]["val"].append(val_loss)
        history["timestamps"].append(time.time())
        print(f"Ep{ep}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    # evaluation on each corpus
    for dname, loader in test_loaders.items():
        acc, swa, cwa, zs_acc, preds, gt = evaluate_test(model, loader)
        print(f"[{dname}] ACC={acc:.4f} SWA={swa:.4f} CWA={cwa:.4f} ZSRTA={zs_acc:.4f}")
        exp_entry = {
            "hidden_dim": hd,
            "metrics": {"test_acc": acc, "SWA": swa, "CWA": cwa, "ZSRTA": zs_acc},
            "predictions": preds,
            "ground_truth": gt,
            "train_curve": history,
        }
        experiment_data["multi_dataset_training"][dname][f"hidden_{hd}"] = exp_entry

# -------------------- save --------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
