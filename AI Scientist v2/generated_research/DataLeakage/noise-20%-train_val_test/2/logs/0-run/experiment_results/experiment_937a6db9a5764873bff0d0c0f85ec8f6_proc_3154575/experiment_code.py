import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------ working dir ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ experiment data --------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------ dataset loader --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded SPR_BENCH:", {k: len(v) for k, v in spr.items()})


# ------------------ tokenisation -----------------
def tokenize(seq: str):
    return list(seq.strip())  # char-level


vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for s in spr["train"]["sequence"]:
    for tok in tokenize(s):
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id, unk_id, cls_id = vocab["<pad>"], vocab["<unk>"], vocab["<cls>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq: str):
    ids = [cls_id] + [vocab.get(t, unk_id) for t in tokenize(seq)]
    return ids


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    maxlen = max(lengths)
    padded = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = maxlen - len(ids)
        padded.append(
            torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
    input_ids = torch.stack(padded)
    attention_mask = (input_ids != pad_id).long()
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ------------------ model ------------------------
class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        n_classes=len(set(spr["train"]["label"])),
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.randn(512, d_model))  # max 512 seq
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids) + self.pos[: input_ids.size(1)]
        if attention_mask is not None:
            # Transformer expects bool mask: True for positions to ignore
            key_padding_mask = ~(attention_mask.bool())
        else:
            key_padding_mask = None
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls_vec = h[:, 0]  # [CLS]
        return self.cls(cls_vec)


model = MiniTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

# ------------------ training loop ---------------
EPOCHS = 5
best_val_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    running_loss = 0.0
    preds, gts = [], []
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(gts, preds, average="macro")
    # ---- validate ----
    model.eval()
    val_loss_tot = 0.0
    v_preds, v_gts = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss_tot += loss.item() * batch["labels"].size(0)
            v_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            v_gts.extend(batch["labels"].cpu().tolist())
    val_loss = val_loss_tot / len(val_loader.dataset)
    val_f1 = f1_score(v_gts, v_preds, average="macro")
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
    )
    # store
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    # keep best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))

# ------------------ test evaluation -------------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.eval()
t_preds, t_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        t_preds.extend(torch.argmax(logits, 1).cpu().tolist())
        t_gts.extend(batch["labels"].cpu().tolist())
test_f1 = f1_score(t_gts, t_preds, average="macro")
print(f"Test macro_f1 = {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = t_preds
experiment_data["SPR_BENCH"]["ground_truth"] = t_gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
