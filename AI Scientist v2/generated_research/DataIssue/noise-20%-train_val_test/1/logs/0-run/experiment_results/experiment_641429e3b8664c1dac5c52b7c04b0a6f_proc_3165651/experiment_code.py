import os, pathlib, numpy as np, torch, random, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# working dir / device --------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment data container ---------------------------------------------------#
experiment_data = {
    "token_level_transformer": {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -----------------------------------------------------------------------------#
# helper to locate dataset ----------------------------------------------------#
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
    raise FileNotFoundError("SPR_BENCH not found – set SPR_DATA or SPR_DATASET_PATH")


# -----------------------------------------------------------------------------#
# load dataset ----------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ("train", "dev", "test")})


data_root = _find_spr_bench()
spr = load_spr_bench(data_root)

# -----------------------------------------------------------------------------#
# tokenisation ----------------------------------------------------------------#
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def build_token_vocab(train_split):
    tokens = set()
    for ex in train_split:
        tokens.update(ex["sequence"].strip().split())
    vocab = {PAD_TOKEN: 0, CLS_TOKEN: 1}
    for tok in sorted(tokens):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_token_vocab(spr["train"])
vocab_size = len(vocab)
pad_id = vocab[PAD_TOKEN]
cls_id = vocab[CLS_TOKEN]
print(f"Vocab size (token level): {vocab_size}")


def encode(seq: str):
    ids = [cls_id] + [vocab[tok] for tok in seq.strip().split()]
    return torch.tensor(ids, dtype=torch.long)


max_len = max(
    len(ex["sequence"].strip().split()) + 1 for ex in spr["train"]
)  # +1 for CLS
print(f"Max token length (with CLS): {max_len}")


# -----------------------------------------------------------------------------#
# Dataset / collate -----------------------------------------------------------#
class SPRTokenDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": encode(row["sequence"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def collate_fn(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn_mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": attn_mask, "labels": labels}


train_ds = SPRTokenDataset(spr["train"])
dev_ds = SPRTokenDataset(spr["dev"])
test_ds = SPRTokenDataset(spr["test"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

num_labels = len(set(int(ex["label"]) for ex in spr["train"]))
print(f"Number of classes: {num_labels}")


# -----------------------------------------------------------------------------#
# Macro-F1 utility ------------------------------------------------------------#
def macro_f1_score(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp == 0 or fn == 0):
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# -----------------------------------------------------------------------------#
# Model -----------------------------------------------------------------------#
class TokenTransformerClassifier(nn.Module):
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
        mlm_prob=0.15,
    ):
        super().__init__()
        self.mlm_prob = mlm_prob
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.cls_head = nn.Linear(d_model, num_labels)
        # MLM head shares weights with embedding
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, compute_mlm=False):
        seq_len = input_ids.size(1)
        x = self.embed(input_ids) + self.pos_embed[:seq_len].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        cls_rep = x[:, 0]  # CLS position
        logits_cls = self.cls_head(self.dropout(cls_rep))
        if compute_mlm:
            mlm_logits = self.mlm_head(self.dropout(x))
            return logits_cls, mlm_logits
        else:
            return logits_cls


# -----------------------------------------------------------------------------#
# Training utilities ----------------------------------------------------------#
def mask_tokens(inputs):
    """Randomly mask tokens for MLM objective"""
    inputs = inputs.clone()
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, model.mlm_prob, device=inputs.device)
    special_mask = (inputs == pad_id) | (inputs == cls_id)
    probability_matrix.masked_fill_(special_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # ignore index
    inputs[masked_indices] = vocab[PAD_TOKEN]  # simple mask as PAD
    return inputs, labels


def run_epoch(model, loader, cls_criterion, mlm_criterion=None, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, cls_losses, mlm_losses = 0.0, 0.0, 0.0
    preds_all, gts_all = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if train_mode and mlm_criterion is not None:
            inputs_masked, mlm_labels = mask_tokens(batch["input_ids"])
            logits_cls, logits_mlm = model(
                inputs_masked, batch["attention_mask"], compute_mlm=True
            )
            loss_cls = cls_criterion(logits_cls, batch["labels"])
            loss_mlm = mlm_criterion(
                logits_mlm.view(-1, vocab_size), mlm_labels.view(-1)
            )
            loss = loss_cls + 0.5 * loss_mlm
        else:
            logits_cls = model(batch["input_ids"], batch["attention_mask"])
            loss = cls_criterion(logits_cls, batch["labels"])
            loss_cls, loss_mlm = loss, torch.tensor(0.0, device=device)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        cls_losses += loss_cls.item() * batch["labels"].size(0)
        mlm_losses += loss_mlm.item() * batch["labels"].size(0)
        preds_all.extend(logits_cls.argmax(1).detach().cpu().numpy())
        gts_all.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = macro_f1_score(gts_all, preds_all, num_labels)
    return avg_loss, macro_f1


# -----------------------------------------------------------------------------#
# Instantiate model / losses / optimiser --------------------------------------#
model = TokenTransformerClassifier(vocab_size, num_labels, max_len).to(device)
cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# -----------------------------------------------------------------------------#
# Training loop with early stopping ------------------------------------------#
best_val_f1, patience, epochs, waited = 0.0, 4, 20, 0
for epoch in range(1, epochs + 1):
    t_start = time.time()
    tr_loss, tr_f1 = run_epoch(model, train_loader, cls_loss_fn, mlm_loss_fn, optimizer)
    val_loss, val_f1 = run_epoch(model, dev_loader, cls_loss_fn)
    experiment_data["token_level_transformer"]["SPR_BENCH"]["metrics"][
        "train_f1"
    ].append(tr_f1)
    experiment_data["token_level_transformer"]["SPR_BENCH"]["metrics"]["val_f1"].append(
        val_f1
    )
    experiment_data["token_level_transformer"]["SPR_BENCH"]["losses"]["train"].append(
        tr_loss
    )
    experiment_data["token_level_transformer"]["SPR_BENCH"]["losses"]["val"].append(
        val_loss
    )
    print(
        f"Epoch {epoch}: val_loss = {val_loss:.4f} | Macro-F1 = {val_f1*100:.2f}% | time {time.time()-t_start:.1f}s"
    )
    if val_f1 > best_val_f1 + 1e-4:
        best_val_f1 = val_f1
        waited = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        waited += 1
        if waited >= patience:
            print("Early stopping triggered.")
            break

# -----------------------------------------------------------------------------#
# Evaluation on test ----------------------------------------------------------#
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
test_loss, test_f1 = run_epoch(model, test_loader, cls_loss_fn)
print(f"TEST Macro-F1: {test_f1*100:.2f}%")

# gather predictions for saving
model.eval()
preds_all, gts_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds_all.extend(logits.argmax(1).cpu().numpy())
        gts_all.extend(batch["labels"].cpu().numpy())
experiment_data["token_level_transformer"]["SPR_BENCH"]["predictions"] = preds_all
experiment_data["token_level_transformer"]["SPR_BENCH"]["ground_truth"] = gts_all

# -----------------------------------------------------------------------------#
# save experiment data --------------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
