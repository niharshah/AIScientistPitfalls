import os, pathlib, time, random, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# reproducibility ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# experiment container -------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_loss": [], "URA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ----------------------------- data utils -----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname: str):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def rule_pattern(sequence: str) -> tuple:
    # canonical multiset representation: sorted token list stored as tuple
    tokens = sequence.strip().split()
    return tuple(sorted(tokens))


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


# ------------------------------ load data -----------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# vocab / label maps ---------------------------------------------------------
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

train_labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(train_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("#labels:", num_labels)

# cached training rule patterns ---------------------------------------------
training_patterns = set(rule_pattern(s) for s in spr["train"]["sequence"])


# -------------------------- torch Dataset -----------------------------------
def encode_seq(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]


class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id, train_mode=True):
        self.raw_seqs = split["sequence"]
        self.encoded = [encode_seq(s, vocab) for s in self.raw_seqs]
        self.labels = split["label"]
        self.train = train_mode
        self.label2id = label2id

    def __len__(self):
        return len(self.raw_seqs)

    def __getitem__(self, idx):
        sample = {
            "input": torch.tensor(self.encoded[idx], dtype=torch.long),
            "raw_seq": self.raw_seqs[idx],
        }
        if self.train:
            sample["label"] = torch.tensor(
                self.label2id[self.labels[idx]], dtype=torch.long
            )
        else:
            sample["label_str"] = self.labels[idx]
        return sample


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    out = {"input": xs_pad, "lengths": lens, "raw_seq": [b["raw_seq"] for b in batch]}
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


# --------------------------- model ------------------------------------------
class MeanPoolSPRModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lin = nn.Linear(emb_dim, num_labels)

    def forward(self, x, lengths):
        emb = self.emb(x)  # [B,T,E]
        mask = (x != 0).unsqueeze(-1).float()  # [B,T,1]
        summed = (emb * mask).sum(1)  # [B,E]
        mean = summed / lengths.clamp(min=1).unsqueeze(-1).float()
        return self.lin(mean)  # [B,C]


# ---------------------- helpers ---------------------------------------------
def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def unseen_rule_accuracy(seqs, preds, golds, train_patterns):
    idx = [i for i, s in enumerate(seqs) if rule_pattern(s) not in train_patterns]
    if not idx:
        return float("nan")
    return np.mean([preds[i] == golds[i] for i in idx])


def run_train_epoch(model, loader, criterion, optimizer):
    model.train()
    tot_loss = tot_acc = tot = 0
    for batch in loader:
        # move tensors -------------------------------------------------------
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = batch["input"].size(0)
        tot_loss += loss.item() * bs
        tot_acc += (logits.argmax(1) == batch["label"]).sum().item()
        tot += bs
    return tot_loss / tot, tot_acc / tot


@torch.no_grad()
def evaluate(model, loader, criterion, train_patterns):
    model.eval()
    tot_loss = tot_acc = tot = 0
    all_seqs = []
    all_preds = []
    all_gold = []
    for batch in loader:
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch_gpu["input"], batch_gpu["lengths"])
        if "label" in batch_gpu:  # dev
            loss = criterion(logits, batch_gpu["label"])
            tot_loss += loss.item() * batch_gpu["input"].size(0)
            tot_acc += (logits.argmax(1) == batch_gpu["label"]).sum().item()
        else:  # test
            loss = torch.tensor(0.0)
        # collect predictions ------------------------------------------------
        preds = logits.argmax(1).cpu().tolist()
        if "label" in batch:
            gold = batch_gpu["label"].cpu().tolist()
            gold = [id2label[g] for g in gold]
        else:
            gold = batch["label_str"]
        all_preds.extend([id2label[p] for p in preds])
        all_gold.extend(gold)
        all_seqs.extend(batch["raw_seq"])
        tot += batch_gpu["input"].size(0)
    acc = (
        tot_acc / tot
        if tot_acc
        else np.mean([p == g for p, g in zip(all_preds, all_gold)])
    )
    ura = unseen_rule_accuracy(all_seqs, all_preds, all_gold, train_patterns)
    return tot_loss / tot if tot else 0.0, acc, ura, all_preds, all_gold, all_seqs


# ------------------------- training loop ------------------------------------
EPOCHS = 3
EMB_DIM = 128

model = MeanPoolSPRModel(len(vocab), EMB_DIM, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_ura, _, _, _ = evaluate(
        model, dev_loader, criterion, training_patterns
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["URA"].append(val_ura)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(
        f"Epoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
        f"val_loss={val_loss:.4f} URA={val_ura:.4f}"
    )

# ------------------------------ final test ----------------------------------
test_loss, test_acc, test_ura, preds, gold, seqs = evaluate(
    model, test_loader, criterion, training_patterns
)
swa = shape_weighted_accuracy(seqs, gold, preds)
cwa = color_weighted_accuracy(seqs, gold, preds)
print(f"TEST  Acc={test_acc:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | URA={test_ura:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gold
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
