import os, pathlib, math, time, json, random, numpy as np, torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ------------------------- mandatory working dir ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "spr_bench": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
# ------------------------- gpu / device -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------- dataset loading ----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


data_path_candidates = [
    pathlib.Path(os.getcwd()) / "SPR_BENCH",
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in data_path_candidates:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found in expected locations.")
spr_bench = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr_bench.items()})


# ------------------------- tokenisation -------------------------------------
def tokenize(seq: str):
    return seq.split(" ") if " " in seq else list(seq.strip())


# Build vocabulary from training data
vocab = {"<PAD>": 0, "<UNK>": 1}
for ex in spr_bench["train"]["sequence"]:
    for tok in tokenize(ex):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Label mapping
labels = sorted(set(spr_bench["train"]["label"]))
label2id = {lab: i for i, lab in enumerate(labels)}
num_classes = len(labels)
print(f"Number of classes: {num_classes}")

# Determine max sequence length
max_len = max(len(tokenize(seq)) for seq in spr_bench["train"]["sequence"])
print(f"Max sequence length: {max_len}")


# ------------------------- Dataset wrapper ----------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]["sequence"]
        label = self.data[idx]["label"]
        toks = tokenize(seq)[:max_len]
        ids = [vocab.get(t, vocab["<UNK>"]) for t in toks]
        attn = [1] * len(ids)
        # pad
        pad_len = max_len - len(ids)
        ids.extend([vocab["<PAD>"]] * pad_len)
        attn.extend([0] * pad_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.float),
            "labels": torch.tensor(label2id[label], dtype=torch.long),
        }


train_ds = SPRTorchDataset(spr_bench["train"])
val_ds = SPRTorchDataset(spr_bench["dev"])
test_ds = SPRTorchDataset(spr_bench["test"])


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------- model -------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_heads, num_layers, num_classes, max_len
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.posenc = PositionalEncoding(emb_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.posenc(x)
        # generate key_padding_mask
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # mean pool over valid tokens
        masked_x = x * attention_mask.unsqueeze(-1)
        sum_x = masked_x.sum(dim=1)
        len_x = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
        pooled = sum_x / len_x
        return self.classifier(pooled)


model = TransformerClassifier(vocab_size, 128, 4, 2, num_classes, max_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------- training loop ------------------------------------
epochs = 5
best_val_f1 = -1
for epoch in range(1, epochs + 1):
    model.train()
    train_losses, train_preds, train_trues = [], [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        train_trues.extend(batch["labels"].cpu().tolist())
    train_f1 = f1_score(train_trues, train_preds, average="macro")

    # validation
    model.eval()
    val_losses, val_preds, val_trues = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_losses.append(loss.item())
            val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            val_trues.extend(batch["labels"].cpu().tolist())
    val_loss = np.mean(val_losses)
    val_f1 = f1_score(val_trues, val_preds, average="macro")

    # logging
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
    )
    experiment_data["spr_bench"]["losses"]["train"].append(float(np.mean(train_losses)))
    experiment_data["spr_bench"]["losses"]["val"].append(float(val_loss))
    experiment_data["spr_bench"]["metrics"]["train_macro_f1"].append(float(train_f1))
    experiment_data["spr_bench"]["metrics"]["val_macro_f1"].append(float(val_f1))
    experiment_data["spr_bench"]["epochs"].append(epoch)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))

# ------------------------- test evaluation ----------------------------------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.eval()
test_preds, test_trues = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        test_trues.extend(batch["labels"].cpu().tolist())
test_macro_f1 = f1_score(test_trues, test_preds, average="macro")
print(f"Test macro_f1: {test_macro_f1:.4f}")
experiment_data["spr_bench"]["predictions"] = test_preds
experiment_data["spr_bench"]["ground_truth"] = test_trues

# ------------------------- save experiment data -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
