import os, pathlib, time, random, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# ---------- reproducibility ----------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------- required working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data ----------
experiment_data = {"d_model_tuning": {"SPR_BENCH": {}}}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- dataset ----------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocabulary ----------
special_tokens = ["<PAD>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size:{vocab_size}  Num classes:{num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[ch] for ch in self.seqs[idx]], dtype=torch.long)
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


# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.cls(pooled)


# ---------- train / eval helpers ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


def train_one_setting(d_model, epochs=5):
    model = SimpleTransformer(
        vocab_size,
        d_model,
        nhead=4,
        num_layers=2,
        num_classes=num_classes,
        pad_id=pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = {"train_loss": [], "val_loss": [], "val_f1": []}
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * batch["label"].size(0)
        train_loss = run_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, dev_loader)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_f1"].append(val_f1)
        print(
            f"d_model={d_model}  Epoch {ep}: train={train_loss:.4f}  "
            f"val={val_loss:.4f}  f1={val_f1:.4f}"
        )
    # final test
    test_loss, test_f1, preds, gts = evaluate(model, test_loader)
    print(f"d_model={d_model}  Test: loss={test_loss:.4f}  f1={test_f1:.4f}\n")
    return metrics, preds, gts, test_loss, test_f1


# ---------- hyperparameter sweep ----------
for d_model in [64, 128, 256, 384]:
    metrics, preds, gts, t_loss, t_f1 = train_one_setting(d_model, epochs=5)
    experiment_data["d_model_tuning"]["SPR_BENCH"][str(d_model)] = {
        "metrics": metrics,
        "predictions": preds,
        "ground_truth": gts,
        "test_loss": t_loss,
        "test_f1": t_f1,
    }

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
