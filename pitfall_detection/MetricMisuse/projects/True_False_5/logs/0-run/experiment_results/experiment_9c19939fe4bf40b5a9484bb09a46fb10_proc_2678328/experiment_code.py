import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------- boiler-plate working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_SWA": [], "val_SWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
exp_rec = experiment_data["SPR_BENCH"]


# ---------- metric helpers (symbolic) ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    wts = [count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(wts, y_true, y_pred)]
    return sum(correct) / (sum(wts) if sum(wts) else 1.0)


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=str(root / split_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- vocabulary ----------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


# ---------- Torch Dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def encode(self, seq):
        return [self.vocab.get(tok, 1) for tok in seq.split()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_str = self.seqs[idx]
        return {
            "input_ids": torch.tensor(self.encode(seq_str), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sym_feats": torch.tensor(
                [
                    count_shape_variety(seq_str),
                    count_color_variety(seq_str),
                    len(seq_str.split()),
                ],
                dtype=torch.float,
            ),
            "sequence_str": seq_str,
        }


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr[spl], vocab) for spl in ["train", "dev", "test"]
)


# ---------- collate ----------
def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    labels = torch.stack([b["labels"] for b in batch])
    sym = torch.stack([b["sym_feats"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "sym_feats": sym,
        "labels": labels,
        "sequence_str": seqs,
    }


B = 128
train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=B, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=B, shuffle=False, collate_fn=collate_fn)


# ---------- Model ----------
class SymbolicTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, ff_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.sym_proj = nn.Linear(3, emb_dim)
        self.classifier = nn.Linear(emb_dim * 2, num_classes)

    def forward(self, ids, mask, sym):
        x = self.emb(ids)
        x = self.enc(x, src_key_padding_mask=(mask == 0))
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1e-6
        )
        sym_emb = torch.relu(self.sym_proj(sym))
        out = torch.cat([pooled, sym_emb], dim=1)
        return self.classifier(out)


num_classes = int(max(train_ds.labels)) + 1
model = SymbolicTransformer(
    len(vocab), emb_dim=64, n_heads=4, ff_dim=128, num_classes=num_classes
).to(device)

# ---------- training setup ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(
                batch_t["input_ids"], batch_t["attention_mask"], batch_t["sym_feats"]
            )
            loss = criterion(logits, batch_t["labels"])
            tot_loss += loss.item() * batch_t["labels"].size(0)
            pred = logits.argmax(-1).cpu().tolist()
            gts.extend(batch["labels"].tolist())
            preds.extend(pred)
            seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return tot_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- training loop ----------
max_epochs = 25
patience = 3
best_val_swa = -1.0
no_improve = 0
best_state = None
for epoch in range(1, max_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch_t["input_ids"], batch_t["attention_mask"], batch_t["sym_feats"]
        )
        loss = criterion(logits, batch_t["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_t["labels"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    train_loss_eval, train_swa, *_ = evaluate(train_loader)
    val_loss, val_swa, *_ = evaluate(dev_loader)
    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_SWA"].append(train_swa)
    exp_rec["metrics"]["val_SWA"].append(val_swa)
    exp_rec["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  val_SWA={val_swa:.4f}")
    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test evaluation ----------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, _ = evaluate(test_loader)
print(f"TEST  loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
