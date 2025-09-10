# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, time, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- reproducibility --------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# -------------------- data store --------------------
experiment_data = {
    "epochs_tuning": {
        "SPR_BENCH": {
            "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
# -------------------- paths / device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset helpers --------------------
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
        d_name = "dev" if split == "dev" else split
        d[d_name] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab --------------------
special_tokens = ["<PAD>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, vocab_size = stoi["<PAD>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, Num classes={num_classes}")


# -------------------- torch dataset --------------------
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
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)


# -------------------- model --------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, n_layers, n_cls, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.cls = nn.Linear(d_model, n_cls)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.cls(pooled)


model = SimpleTransformer(vocab_size, 128, 4, 2, num_classes, pad_id).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------- evaluation --------------------
def evaluate(loader):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], mask)
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


# -------------------- training w/ early stopping --------------------
max_epochs, patience = 30, 5
best_f1, epochs_no_improve = -1.0, 0

for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["input_ids"] == pad_id)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)
    train_loss = running_loss / len(train_loader.dataset)

    val_loss, val_f1, _, _ = evaluate(dev_loader)
    print(
        f"E{epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
    )

    ed = experiment_data["epochs_tuning"]["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)

    # -------- early stopping --------
    if val_f1 > best_f1:
        best_f1, epochs_no_improve = val_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# -------------------- load best model & test --------------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = evaluate(test_loader)
print(f"Test : loss={test_loss:.4f}  macro_f1={test_f1:.4f}")

ed["predictions"], ed["ground_truth"] = test_preds, test_gts

# -------------------- save --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
