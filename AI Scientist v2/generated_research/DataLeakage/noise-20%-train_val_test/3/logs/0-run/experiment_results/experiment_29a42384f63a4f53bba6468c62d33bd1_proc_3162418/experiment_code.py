import os, pathlib, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- setup / reproducibility ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- dataset load ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    dd["train"] = _load("train.csv")
    dd["dev"] = _load("dev.csv")
    dd["test"] = _load("test.csv")
    return dd


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})

# ---------- vocab ----------
special_tokens = ["<PAD>", "<CLS>"]
chars = set(ch for s in dsets["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
PAD_ID, CLS_ID = stoi["<PAD>"], stoi["<CLS>"]
vocab_size = len(itos)
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab:{vocab_size}, Classes:{num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_ids = [CLS_ID] + [stoi[c] for c in self.seqs[idx]]
        ids = torch.tensor(seq_ids, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # symbol counts (exclude CLS)
        counts = torch.bincount(
            torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long),
            minlength=vocab_size,
        ).float()
        return {"input_ids": ids, "label": label, "counts": counts}


def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    counts = torch.stack([b["counts"] for b in batch])
    padded = pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    return {"input_ids": padded, "label": labels, "counts": counts}


train_loader = DataLoader(
    SPRTorchDataset(dsets["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(dsets["dev"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(dsets["test"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class HybridTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, num_cls, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.count_proj = nn.Sequential(
            nn.Linear(vocab, d_model), nn.ReLU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_cls),
        )

    def forward(self, ids, counts):
        mask = ids == PAD_ID
        x = self.embed(ids)
        x = self.encoder(x, src_key_padding_mask=mask)
        cls_repr = x[:, 0]  # CLS position
        cnt_repr = self.count_proj(counts)
        combined = torch.cat([cls_repr, cnt_repr], dim=-1)
        return self.classifier(combined)


model = HybridTransformer(vocab_size, 128, 4, 4, num_classes, PAD_ID).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)


# ---------- evaluation ----------
def run_epoch(loader, train=False):
    ep_loss, preds, gts = 0.0, [], []
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["counts"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ep_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    avg_loss = ep_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- training loop with early stopping ----------
best_f1, patience, no_improve = -1, 6, 0
max_epochs = 30
best_state = None

for epoch in range(1, max_epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, train=False)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
    )
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(tr_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)
    if val_f1 > best_f1:
        best_f1, no_improve = val_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered")
            break

# ---------- test ----------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, train=False)
print(f"Test: loss={test_loss:.4f} macro_f1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
