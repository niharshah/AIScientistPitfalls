import os, pathlib, random, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------
# working dir & reproducibility
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# experiment data holder
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -----------------------------------------------------------------------------
# dataset utilities from provided helper
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# -----------------------------------------------------------------------------
# vocabulary (character level)
special_tokens = ["<PAD>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, vocab_size = stoi["<PAD>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
max_len = max(len(seq) for seq in spr["train"]["sequence"])
print(f"Vocab={vocab_size}   Classes={num_classes}   MaxLen={max_len}")


# -----------------------------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# -----------------------------------------------------------------------------
class HybridTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        num_classes,
        d_model=128,
        nhead=8,
        n_layers=4,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.sym_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.count_ff = nn.Sequential(
            nn.Linear(vocab_size, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.cls = nn.Linear(d_model * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        pad_mask = input_ids == self.pad_id  # (B,L)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(
            0
        )
        x = self.sym_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,L,D)

        # masked mean pooling
        mask = (~pad_mask).unsqueeze(-1).type_as(x)  # (B,L,1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B,D)

        # differentiable histogram (ignore PAD)
        one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()  # (B,L,V)
        one_hot = one_hot * (~pad_mask).unsqueeze(-1)  # zero pad counts
        counts = one_hot.sum(1)  # (B,V)
        count_vec = self.count_ff(counts)  # (B,D)

        feat = torch.cat([pooled, count_vec], dim=-1)  # (B,2D)
        logits = self.cls(self.dropout(feat))  # (B,C)
        return logits


model = HybridTransformer(vocab_size, pad_id, num_classes, max_len=max_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


# -----------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -----------------------------------------------------------------------------
max_epochs, patience = 30, 5
best_f1, patience_ctr = -1.0, 0
best_state = None

for epoch in range(1, max_epochs + 1):
    # training --------------------------------------------------------
    model.train()
    running = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["label"].size(0)
    train_loss = running / len(train_loader.dataset)

    # validation ------------------------------------------------------
    val_loss, val_f1, _, _ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  Macro-F1 = {val_f1:.4f}")

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)

    # early stopping --------------------------------------------------
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("Early stopping.")
            break

# -----------------------------------------------------------------------------
# test set evaluation with best model
model.load_state_dict(best_state)
model.to(device)
test_loss, test_f1, test_preds, test_gts = evaluate(test_loader)
print(f"Test results -> loss: {test_loss:.4f}  Macro-F1: {test_f1:.4f}")

ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# -----------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
