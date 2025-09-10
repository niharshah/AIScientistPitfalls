import os, pathlib, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# experiment data container ---------------------------------------------------#
experiment_data = {
    "contextual_big_transformer": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "test_acc": None,
            "test_f1": None,
            "test_loss": None,
        }
    }
}

# -----------------------------------------------------------------------------#
# directories / device --------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------------------------#
# locate SPR_BENCH ------------------------------------------------------------#
def _find_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path(os.getenv("SPR_DATASET_PATH", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]
    files = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and files.issubset({p.name for p in c.iterdir()}):
            print(f"Found SPR_BENCH at: {c}")
            return c
    raise FileNotFoundError(
        "Could not locate SPR_BENCH.  Set SPR_DATA or SPR_DATASET_PATH env var."
    )


# -----------------------------------------------------------------------------#
# dataset helpers -------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ("train", "dev", "test")})


CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


class SPRCharDataset(Dataset):
    def __init__(self, hf_dataset, vocab):
        self.data = hf_dataset
        self.vocab = vocab
        self.pad_id = vocab[PAD_TOKEN]
        self.cls_id = vocab[CLS_TOKEN]

    def __len__(self):
        return len(self.data)

    def _encode(self, seq: str):
        seq = seq.replace(" ", "")
        ids = [self.cls_id] + [self.vocab[ch] for ch in seq]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": self._encode(row["sequence"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def build_vocab(train_split):
    chars = {ch for ex in train_split for ch in ex["sequence"].replace(" ", "")}
    vocab = {PAD_TOKEN: 0, CLS_TOKEN: 1}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def collate_fn(batch, pad_id=0):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}


# -----------------------------------------------------------------------------#
# model -----------------------------------------------------------------------#
class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        max_len,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.embed(input_ids) + self.pos_embed[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        cls_rep = x[:, 0]  # representation of CLS token
        return self.classifier(cls_rep)


# -----------------------------------------------------------------------------#
# loss ------------------------------------------------------------------------#
class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        loss = (1 - self.eps) * loss - self.eps * log_probs.mean(dim=-1)
        return loss.mean()


# -----------------------------------------------------------------------------#
# metrics ---------------------------------------------------------------------#
def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_labels: int):
    f1s = []
    for lbl in range(num_labels):
        tp = ((preds == lbl) & (targets == lbl)).sum().item()
        fp = ((preds == lbl) & (targets != lbl)).sum().item()
        fn = ((preds != lbl) & (targets == lbl)).sum().item()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / num_labels


# -----------------------------------------------------------------------------#
# train / eval loops -----------------------------------------------------------#
def run_epoch(model, loader, criterion, num_labels, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    preds_all, gts_all = [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
            preds = outputs.argmax(1)
            preds_all.append(preds.cpu())
            gts_all.append(batch["labels"].cpu())
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    preds_all = torch.cat(preds_all)
    gts_all = torch.cat(gts_all)
    acc = correct / total
    f1 = macro_f1(preds_all, gts_all, num_labels)
    return epoch_loss / total, acc, f1, preds_all, gts_all


# -----------------------------------------------------------------------------#
# main pipeline ---------------------------------------------------------------#
data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
print("Loaded SPR_BENCH splits:", list(spr.keys()))

vocab = build_vocab(spr["train"])
print(f"Vocab size: {len(vocab)}")
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"]) + 1  # +CLS
print(f"Max sequence length (with CLS): {max_len}")

train_ds = SPRCharDataset(spr["train"], vocab)
dev_ds = SPRCharDataset(spr["dev"], vocab)
test_ds = SPRCharDataset(spr["test"], vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab[PAD_TOKEN]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab[PAD_TOKEN]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_id=vocab[PAD_TOKEN]),
)

num_labels = len(set(int(ex["label"]) for ex in spr["train"]))
epochs = 8
criterion = LabelSmoothingCE(eps=0.1)

model = SimpleTransformerClassifier(
    len(vocab), num_labels, max_len=max_len, dropout=0.1
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

best_val_f1, best_state = 0.0, None
for epoch in range(1, epochs + 1):
    t_loss, t_acc, t_f1, _, _ = run_epoch(
        model, train_loader, criterion, num_labels, optimizer
    )
    v_loss, v_acc, v_f1, _, _ = run_epoch(
        model, dev_loader, criterion, num_labels, optimizer=None
    )
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["metrics"][
        "train_acc"
    ].append(t_acc)
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["metrics"][
        "val_acc"
    ].append(v_acc)
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["metrics"][
        "train_f1"
    ].append(t_f1)
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["metrics"][
        "val_f1"
    ].append(v_f1)
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["losses"][
        "train"
    ].append(t_loss)
    experiment_data["contextual_big_transformer"]["SPR_BENCH"]["losses"]["val"].append(
        v_loss
    )
    print(
        f"Epoch {epoch}: val_loss={v_loss:.4f}, val_acc={v_acc*100:.2f}%, val_macroF1={v_f1:.4f}"
    )
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        best_state = {
            k: v.clone().detach().cpu() for k, v in model.state_dict().items()
        }

# load best model
model.load_state_dict(best_state)
test_loss, test_acc, test_f1, preds_all, gts_all = run_epoch(
    model, test_loader, criterion, num_labels, optimizer=None
)
experiment_data["contextual_big_transformer"]["SPR_BENCH"][
    "predictions"
] = preds_all.numpy()
experiment_data["contextual_big_transformer"]["SPR_BENCH"][
    "ground_truth"
] = gts_all.numpy()
experiment_data["contextual_big_transformer"]["SPR_BENCH"]["test_acc"] = test_acc
experiment_data["contextual_big_transformer"]["SPR_BENCH"]["test_f1"] = test_f1
experiment_data["contextual_big_transformer"]["SPR_BENCH"]["test_loss"] = test_loss

print(
    f"\nBEST MODEL -- Test accuracy: {test_acc*100:.2f}%, Test macro-F1: {test_f1:.4f}"
)

# -----------------------------------------------------------------------------#
# save all experiment data ----------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
