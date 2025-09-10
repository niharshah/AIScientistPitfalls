import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# mandatory working dir & device handling
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -----------------------------------------------------------------------------
# -------------------- DATA ----------------------------------------------------
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    spr_raw = load_spr_bench(DATA_PATH)
except Exception as e:
    print("Dataset not found, generating synthetic SPR data…")

    def _synth(n):
        vocab = [f"tok{i}" for i in range(20)]
        data = []
        for i in range(n):
            seq_len = random.randint(5, 15)
            seq = " ".join(random.choices(vocab, k=seq_len))
            label = random.randint(0, 3)
            data.append({"id": str(i), "sequence": seq, "label": label})
        return data

    import datasets

    spr_raw = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(_synth(1000)),
            "dev": datasets.Dataset.from_list(_synth(200)),
            "test": datasets.Dataset.from_list(_synth(200)),
        }
    )

# -------------------- TOKENISER ----------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"
token_to_idx = {PAD: 0, UNK: 1}
label_to_idx = {}
max_len = 0
for ex in spr_raw["train"]:
    tokens = ex["sequence"].split()
    max_len = max(max_len, len(tokens))
    for tok in tokens:
        if tok not in token_to_idx:
            token_to_idx[tok] = len(token_to_idx)
    lbl = ex["label"]
    if lbl not in label_to_idx:
        label_to_idx[lbl] = len(label_to_idx)
vocab_size = len(token_to_idx)
num_classes = len(label_to_idx)
print(f"Vocab size={vocab_size}, classes={num_classes}, max_len={max_len}")


def encode_sequence(text):
    ids = [token_to_idx.get(tok, token_to_idx[UNK]) for tok in text.split()]
    if len(ids) < max_len:
        ids += [token_to_idx[PAD]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def encode_label(l):
    return label_to_idx[l]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": torch.tensor(
                encode_sequence(row["sequence"]), dtype=torch.long
            ),
            "label": torch.tensor(encode_label(row["label"]), dtype=torch.long),
        }


batch_size = 64
datasets_torch = {k: SPRTorchDataset(v) for k, v in spr_raw.items()}
loaders = {
    k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"))
    for k, v in datasets_torch.items()
}


# -------------------- MODEL ---------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, nhead=4, num_layers=2, n_classes=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=token_to_idx[PAD])
        self.pos = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids):
        mask = input_ids == token_to_idx[PAD]
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        # masked mean
        mask_inv = (~mask).unsqueeze(-1).float()
        x = (x * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1.0)
        return self.classifier(x)


model = SPRTransformer(vocab_size, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------- METRICS STORAGE ----------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------- TRAIN LOOP ---------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    # TRAIN
    model.train()
    train_losses = []
    train_preds, train_gts = [], []
    for batch in loaders["train"]:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_preds.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
        train_gts.extend(batch["label"].cpu().numpy().tolist())
    train_f1 = f1_score(train_gts, train_preds, average="macro")
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"epoch": epoch, "Macro_F1": train_f1}
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(
        {"epoch": epoch, "loss": np.mean(train_losses)}
    )

    # VALIDATION
    model.eval()
    val_losses = []
    val_preds, val_gts = [], []
    with torch.no_grad():
        for batch in loaders["dev"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            val_losses.append(loss.item())
            val_preds.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
            val_gts.extend(batch["label"].cpu().numpy().tolist())
    val_f1 = f1_score(val_gts, val_preds, average="macro")
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "Macro_F1": val_f1}
    )
    experiment_data["SPR_BENCH"]["losses"]["val"].append(
        {"epoch": epoch, "loss": np.mean(val_losses)}
    )
    print(
        f"Epoch {epoch}: validation_loss = {np.mean(val_losses):.4f}, Macro_F1 = {val_f1:.3f}"
    )

# -------------------- TEST EVAL ----------------------------------------------
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in loaders["test"]:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        test_preds.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
        test_gts.extend(batch["label"].cpu().numpy().tolist())
test_f1 = f1_score(test_gts, test_preds, average="macro")
print(f"Test Macro_F1 = {test_f1:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# -------------------- SAVE ----------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
