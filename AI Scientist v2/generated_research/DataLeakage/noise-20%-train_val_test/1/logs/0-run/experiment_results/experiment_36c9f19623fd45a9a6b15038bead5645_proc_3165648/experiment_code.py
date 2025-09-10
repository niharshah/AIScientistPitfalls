import os, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# required working dir and device                                              #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment data container                                                    #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------------------------#
# locate and load dataset                                                      #
def _find_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]
    needed = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and needed.issubset({p.name for p in c.iterdir()}):
            print(f"Found SPR_BENCH at {c}")
            return c
    raise FileNotFoundError(
        "SPR_BENCH not found; set SPR_DATA env var or place folder."
    )


data_root = _find_spr_bench()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=str(root / split_name),
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


spr = load_spr_bench(data_root)


# -----------------------------------------------------------------------------#
# vocabulary                                                                   #
def build_vocab(train_split):
    vocab = {"<pad>": 0, "<cls>": 1}
    for ex in train_split:
        for tok in ex["sequence"].strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print(f"Vocabulary size: {len(vocab)}")


# -----------------------------------------------------------------------------#
# dataset object                                                               #
class SPRTokenDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.data = hf_dataset
        self.vocab = vocab
        self.cls_id = vocab["<cls>"]
        self.pad_id = vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _encode(self, seq: str):
        toks = seq.strip().split()
        return [self.cls_id] + [self.vocab[t] for t in toks]

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(self._encode(row["sequence"]), dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        length = len(row["sequence"].split())
        parity = torch.tensor(length % 2, dtype=torch.long)
        distinct_ct = min(len(set(row["sequence"].split())), 10)  # bucket 0-10+
        distinct = torch.tensor(distinct_ct, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": label,
            "parity": parity,
            "distinct": distinct,
        }


def collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn = (padded != pad_id).long()
    labels = torch.stack([b["labels"] for b in batch])
    parity = torch.stack([b["parity"] for b in batch])
    distinct = torch.stack([b["distinct"] for b in batch])
    return {
        "input_ids": padded,
        "attention_mask": attn,
        "labels": labels,
        "parity": parity,
        "distinct": distinct,
    }


train_ds = SPRTokenDataset(spr["train"], vocab)
dev_ds = SPRTokenDataset(spr["dev"], vocab)
test_ds = SPRTokenDataset(spr["test"], vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)

num_rule_labels = len({int(r["label"]) for r in spr["train"]})
num_parity_labels = 2
num_distinct_labels = 11
max_len = max(len(r["sequence"].split()) + 1 for r in spr["train"])
print(f"Max token length: {max_len}")


# -----------------------------------------------------------------------------#
# model                                                                        #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[:, : x.size(1)]


class SPRMultiHeadTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model=160, nhead=8, nlayers=6, ff=320, dropout=0.2
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.main_head = nn.Linear(d_model, num_rule_labels)
        self.parity_head = nn.Linear(d_model, num_parity_labels)
        self.distinct_head = nn.Linear(d_model, num_distinct_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=(attention_mask == 0))
        cls = self.norm(x[:, 0])
        return (self.main_head(cls), self.parity_head(cls), self.distinct_head(cls))


model = SPRMultiHeadTransformer(len(vocab)).to(device)


# -----------------------------------------------------------------------------#
# loss functions & optimizer                                                   #
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n = logits.size(1)
        log_prob = nn.functional.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_prob, dim=1))


main_criterion = LabelSmoothingCE(0.1)
aux_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


# -----------------------------------------------------------------------------#
# metrics                                                                      #
def macro_f1(preds, labels, num_cls):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    f1s = []
    for c in range(num_cls):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        if tp == fp == fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        f1s.append(f1)
    return float(np.mean(f1s))


# -----------------------------------------------------------------------------#
# training & evaluation loops                                                  #
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_cnt = 0.0
    all_preds, all_lbls = [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        main_logits, parity_logits, distinct_logits = model(
            batch["input_ids"], batch["attention_mask"]
        )
        loss_main = main_criterion(main_logits, batch["labels"])
        loss_par = aux_criterion(parity_logits, batch["parity"])
        loss_dist = aux_criterion(distinct_logits, batch["distinct"])
        loss = loss_main + 0.2 * loss_par + 0.2 * loss_dist
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = main_logits.argmax(1)
        tot_loss += loss.item() * len(preds)
        tot_correct += (preds == batch["labels"]).sum().item()
        tot_cnt += len(preds)
        all_preds.append(preds)
        all_lbls.append(batch["labels"])
    all_preds = torch.cat(all_preds)
    all_lbls = torch.cat(all_lbls)
    acc = tot_correct / tot_cnt
    f1 = macro_f1(all_preds, all_lbls, num_rule_labels)
    return tot_loss / tot_cnt, acc, f1


# -----------------------------------------------------------------------------#
# main training loop                                                           #
epochs = 6
best_f1 = 0.0
for ep in range(1, epochs + 1):
    t_loss, t_acc, t_f1 = run_epoch(train_loader, train=True)
    v_loss, v_acc, v_f1 = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(t_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(t_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(v_acc)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(t_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(v_f1)
    print(f"Epoch {ep}: validation_loss = {v_loss:.4f}, val_macroF1 = {v_f1:.4f}")
    # simple checkpoint
    if v_f1 > best_f1:
        best_f1 = v_f1
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))

# -----------------------------------------------------------------------------#
# test evaluation with best checkpoint                                         #
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
model.eval()
with torch.no_grad():
    preds_all, labels_all = [], []
    for batch in test_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits, _, _ = model(batch["input_ids"], batch["attention_mask"])
        preds_all.append(logits.argmax(1).cpu())
        labels_all.append(batch["labels"].cpu())
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
test_acc = (preds_all == labels_all).float().mean().item()
test_f1 = macro_f1(preds_all, labels_all, num_rule_labels)
print(f"Test accuracy: {test_acc*100:.2f}%, Test macroF1: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds_all.numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = labels_all.numpy()

# -----------------------------------------------------------------------------#
# save experiment data                                                         #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
