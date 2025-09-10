import os, pathlib, time, random, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Tuple, Any

# -------------------------------------------------------------------
# working directory & container for logs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "val_loss": [],
            "URA_val": [],
            "URA_test": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
# -------------------------------------------------------------------
# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------
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


# ---------------- vocabulary ------------------------------------------------
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

# ---------------- labels ----------------------------------------------------
train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("#labels:", num_labels)


# ---------------- helper functions -----------------------------------------
def encode_seq(seq: str, vocab_: Dict[str, int]) -> List[int]:
    return [vocab_.get(tok, vocab_["<unk>"]) for tok in seq.strip().split()]


def pattern_of(seq: str) -> Tuple[str, ...]:
    """Return a canonical tuple representing shape–color multiset."""
    return tuple(sorted(seq.strip().split()))


train_patterns = set(pattern_of(s) for s in spr["train"]["sequence"])


def unseen_mask(sequences: List[str], seen_patterns: set) -> List[bool]:
    return [pattern_of(s) not in seen_patterns for s in sequences]


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


# ---------------- PyTorch dataset ------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(
        self, hf_split, vocab: Dict[str, int], label2id_: Dict[str, int], train: bool
    ):
        self.seqs_tok = [encode_seq(s, vocab) for s in hf_split["sequence"]]
        self.seqs_raw = hf_split["sequence"]
        self.labels_raw = hf_split["label"]
        self.train = train
        self.label2id_ = label2id_

    def __len__(self):
        return len(self.seqs_tok)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs_tok[idx], dtype=torch.long)
        out = {"input": x, "raw_seq": self.seqs_raw[idx]}
        if self.train:
            out["label"] = torch.tensor(
                self.label2id_[self.labels_raw[idx]], dtype=torch.long
            )
        else:
            out["label_str"] = self.labels_raw[idx]
        return out


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [b["input"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": lens, "raw_seq": [b["raw_seq"] for b in batch]}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], vocab, label2id, train=True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, train=True)
test_ds = SPRTorchDataset(spr["test"], vocab, label2id, train=False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------------- model -----------------------------------------------------
class MeanPoolSPRModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_labels_: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_labels_)

    def forward(self, x, lengths):
        e = self.emb(x)  # [B,T,E]
        mask = (x != 0).unsqueeze(-1).float()  # [B,T,1]
        summed = (e * mask).sum(1)  # [B,E]
        mean = summed / lengths.clamp(min=1).unsqueeze(-1).float()
        return self.fc(mean)


# ------------------- evaluation helpers ------------------------------------
@torch.no_grad()
def evaluate(loader: DataLoader, model: nn.Module) -> Tuple[float, float]:
    model.eval()
    all_pred, all_gold, all_seq = [], [], []
    for batch in loader:
        # move tensors to device
        batch_t = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch_t["input"], batch_t["lengths"])
        preds = logits.argmax(1).cpu().tolist()
        seqs = batch["raw_seq"]
        all_seq.extend(seqs)
        all_pred.extend([id2label[p] for p in preds])
        if "label" in batch:  # dev/val
            gold = [id2label[i.item()] for i in batch["label"]]
        else:  # test
            gold = batch["label_str"]
        all_gold.extend(gold)
    acc = np.mean([p == g for p, g in zip(all_pred, all_gold)])
    mask = unseen_mask(all_seq, train_patterns)
    if any(mask):
        ura = np.mean([p == g for p, g, m in zip(all_pred, all_gold, mask) if m])
    else:
        ura = float("nan")
    return acc, ura


# ------------------- training utilities ------------------------------------
def run_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer=None
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in loader:
        # move tensors to device
        batch_t = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch_t["input"], batch_t["lengths"])
        loss = criterion(logits, batch_t["label"])
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch_t["input"].size(0)
        total_correct += (logits.argmax(1) == batch_t["label"]).sum().item()
        total_count += batch_t["input"].size(0)
    return total_loss / total_count, total_correct / total_count


# -------------------- training loop ----------------------------------------
EPOCHS = 5
EMB_DIM = 128
model = MeanPoolSPRModel(len(vocab), EMB_DIM, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, dev_loader, criterion)
    _, ura_val = evaluate(dev_loader, model)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["URA_val"].append(ura_val)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: "
        f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
        f"val_loss={val_loss:.4f} URA_val={ura_val:.4f}"
    )

# -------------------- final test evaluation ---------------------------------
test_acc, ura_test = evaluate(test_loader, model)
experiment_data["SPR_BENCH"]["metrics"]["URA_test"].append(ura_test)

print(f"TEST Acc={test_acc:.4f} | URA_test={ura_test:.4f}")


# also compute additional metrics required by original script
@torch.no_grad()
def detailed_test(model):
    model.eval()
    preds, gold, seqs = [], [], []
    for batch in test_loader:
        bt = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(bt["input"], bt["lengths"])
        p = logits.argmax(1).cpu().tolist()
        preds.extend([id2label[i] for i in p])
        gold.extend(batch["label_str"])
        seqs.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(seqs, gold, preds)
    cwa = color_weighted_accuracy(seqs, gold, preds)
    return swa, cwa, preds, gold


swa, cwa, preds, gold = detailed_test(model)
print(f"SWA={swa:.4f} | CWA={cwa:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gold

# -------------------- save everything --------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
