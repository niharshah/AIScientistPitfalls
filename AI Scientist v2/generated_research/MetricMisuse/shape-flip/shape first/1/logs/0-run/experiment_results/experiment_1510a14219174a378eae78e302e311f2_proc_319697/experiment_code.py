# ------------------- zero_shot_spr_meanpool_ura.py -------------------
import os, pathlib, time, random, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------- reproducibility -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------- data loading ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ----------------------------- vocab ---------------------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
idx2tok = [None] * len(vocab)
for tok, idx in vocab.items():
    idx2tok[idx] = tok
print("Vocab size:", len(vocab))


def encode_seq(seq: str, vocab) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("#labels:", num_labels)


# -------------------------- datasets ---------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id, train_mode=True):
        self.seqs = [encode_seq(s, vocab) for s in split["sequence"]]
        self.seq_str = split["sequence"]
        self.labels = split["label"]
        self.train = train_mode

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        sample = {"input": x, "seq_str": self.seq_str[idx]}
        if self.train:
            sample["label"] = torch.tensor(label2id[self.labels[idx]], dtype=torch.long)
        else:
            sample["label_str"] = self.labels[idx]
        return sample


def collate_fn(batch):
    xs = [b["input"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": lens, "seq_str": [b["seq_str"] for b in batch]}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], vocab, label2id, True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, True)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id, False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)


# -------------------------- helpers ----------------------------------
def seq_signature(sequence: str) -> tuple:
    """Canonical multiset signature of tokens (sorted keeps multiplicity)."""
    return tuple(sorted(sequence.strip().split()))


train_signatures = set(seq_signature(s) for s in spr["train"]["sequence"])


def unseen_rule_accuracy(seqs, gold, pred, train_sigs):
    idxs = [i for i, s in enumerate(seqs) if seq_signature(s) not in train_sigs]
    if not idxs:
        return float("nan")
    return np.mean([gold[i] == pred[i] for i in idxs])


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


# --------------------------- model -----------------------------------
class MeanPoolSPR(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.head = nn.Linear(emb_dim, num_labels)

    def forward(self, x, lengths):
        e = self.emb(x)  # [B,T,E]
        mask = (x != 0).unsqueeze(-1).float()  # [B,T,1]
        summed = (e * mask).sum(1)  # [B,E]
        mean = summed / lengths.clamp(min=1).unsqueeze(-1).float()
        return self.head(mean)


# ------------------------ train / eval -------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot = 0
    for batch in loader:
        # move tensors to device
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        x, lens = batch_t["input"], batch_t["lengths"]
        logits = model(x, lens)
        y = batch_t["label"]
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_loss / tot, tot_correct / tot


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_gold, all_seqs = [], [], []
    for batch in loader:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        x, lens = batch_t["input"], batch_t["lengths"]
        logits = model(x, lens)
        preds_idx = logits.argmax(1).cpu().tolist()
        all_preds.extend([id2label.get(i, "UNK") for i in preds_idx])
        all_gold.extend(batch["label_str"])
        all_seqs.extend(batch["seq_str"])
    acc = np.mean([p == g for p, g in zip(all_preds, all_gold)])
    swa = shape_weighted_accuracy(all_seqs, all_gold, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_gold, all_preds)
    ura = unseen_rule_accuracy(all_seqs, all_gold, all_preds, train_signatures)
    return acc, swa, cwa, ura, all_preds, all_gold


# -------------------- experiment container ---------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "URA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------------------- train ----------------------------------
EMB_DIMS = [128]  # keep short for runtime
EPOCHS = 5
criterion = nn.CrossEntropyLoss()

for emb_dim in EMB_DIMS:
    print(f"\n=== Embedding dim {emb_dim} ===")
    model = MeanPoolSPR(len(vocab), emb_dim, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, dev_loader, criterion)  # no opt -> eval
        # compute URA on dev set
        dev_preds, dev_gold, dev_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch_t["input"], batch_t["lengths"])
                preds = logits.argmax(1).cpu().tolist()
                dev_preds.extend([id2label.get(i, "UNK") for i in preds])
                dev_gold.extend(batch["label"].cpu().apply_(lambda i: int(i)).tolist())
                dev_seqs.extend(batch["seq_str"])
        ura_val = unseen_rule_accuracy(
            dev_seqs, [train_labels[i] for i in dev_gold], dev_preds, train_signatures
        )

        experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
        experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["URA"].append(ura_val)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"URA={ura_val if not math.isnan(ura_val) else 'nan'}"
        )

    # --------------- final evaluation on test ------------------------
    acc, swa, cwa, ura, preds, gold = evaluate(model, test_loader)
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gold
    print(f"TEST: Acc={acc:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | URA={ura}")

# --------------------------- saving ----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
