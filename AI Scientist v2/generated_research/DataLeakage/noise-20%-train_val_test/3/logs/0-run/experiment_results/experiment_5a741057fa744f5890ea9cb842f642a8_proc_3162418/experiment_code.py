import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------- experiment storage ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- dataset loading ----------
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
print({k: len(v) for k, v in spr.items()})

# ---------- vocabulary ----------
special_tokens = ["<PAD>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, vocab_size = stoi["<PAD>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"vocab={vocab_size}, classes={num_classes}")


# ---------- torch Datasets ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    # counts per sequence
    cnts = []
    for ids in seqs:
        cnt = torch.bincount(ids, minlength=vocab_size).float()
        cnts.append(cnt)
    cnts = torch.stack(cnts)
    return {"input_ids": padded, "counts": cnts, "label": labels}


BATCH = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class DualPathTransformer(nn.Module):
    def __init__(
        self, vocab, d_model, nhead, nlayers, n_cls, pad_idx, cnt_dim=64, dropout=0.2
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cnt_proj = nn.Sequential(
            nn.Linear(vocab, cnt_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model + cnt_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_cls),
        )

    def forward(self, ids, counts, pad_mask):
        x = self.embed(ids)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        cnt_repr = self.cnt_proj(counts)
        out = self.fc(torch.cat([pooled, cnt_repr], dim=-1))
        return out


model = DualPathTransformer(vocab_size, 128, 4, 4, num_classes, pad_id).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)


# ---------- evaluation ----------
def evaluate(loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(
                batch["input_ids"], batch["counts"], batch["input_ids"] == pad_id
            )
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------- training loop with early stopping ----------
best_f1, best_state, patience, max_epochs = -1.0, None, 6, 40
epochs_no_improve = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"], batch["counts"], batch["input_ids"] == pad_id
        )
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch["label"].size(0)
    train_loss /= len(train_loader.dataset)

    val_loss, val_f1, _, _ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_f1 = {val_f1:.4f}")

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test ----------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = evaluate(test_loader)
print(f"Test: loss={test_loss:.4f}, macro_f1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
