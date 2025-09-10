import os, pathlib, random, string, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# helper to load SPR_BENCH or create synthetic fallback -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file_):
        return load_dataset(
            "csv", data_files=str(root / file_), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def synthetic_spr(samples=2000, max_len=12, n_rules=5):
    vocab = list(string.ascii_uppercase[:10])

    def gen_seq():
        ln = random.randint(4, max_len)
        return "".join(random.choice(vocab) for _ in range(ln))

    data = {"train": [], "dev": [], "test": []}
    for split, n in [("train", samples), ("dev", samples // 4), ("test", samples // 2)]:
        for i in range(n):
            seq = gen_seq()
            label = random.randint(0, n_rules - 1)
            data[split].append({"id": i, "sequence": seq, "label": label})
    dsd = DatasetDict(
        {
            k: load_dataset("json", data_files={k: data_list}, split=k)
            for k, data_list in {k: f"{k}.json" for k in data}.items()
        }
    )
    # Actually easier: make datasets from lists directly
    return DatasetDict(
        {
            k: load_dataset(
                "json",
                data_files={k: [os.path.join(working_dir, f"{k}.json") for _ in [0]]},
                split=k,
            )
            for k in ["train", "dev", "test"]
        }
    )


# Try loading real benchmark
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH from disk.")
except Exception as e:
    print("Could not load SPR_BENCH, creating synthetic toy data.", e)
    # build synthetic DatasetDict manually
    splits = {}
    vocab = list(string.ascii_uppercase[:10])

    def make_split(n):
        seqs, labels, ids = [], [], []
        for i in range(n):
            ln = random.randint(4, 12)
            seqs.append("".join(random.choice(vocab) for _ in range(ln)))
            labels.append(random.randint(0, 4))
            ids.append(i)
        return {"id": ids, "sequence": seqs, "label": labels}

    for sp, n in [("train", 2000), ("dev", 500), ("test", 1000)]:
        splits[sp] = load_dataset(
            "json", data_files={sp: []}, split="train"
        )  # placeholder
    # Instead, circumvent HF; fallback will be built later
    raise SystemExit("Synthetic generation not implemented for HF fallback.")

# ------------------------------------------------------------------
# Build vocabulary -------------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab_chars = {ch for seq in spr["train"]["sequence"] for ch in seq}
itos = [PAD, UNK] + sorted(vocab_chars)
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [stoi.get(ch, stoi[UNK]) for ch in seq]


# ------------------------------------------------------------------
# Torch Dataset ----------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = [encode(s) for s in hf_ds["sequence"]]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seqs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    input_lens = [len(x["input_ids"]) for x in batch]
    max_len = max(input_lens)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        ids = b["input_ids"]
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_ds = SPRDataset(spr["train"])
dev_ds = SPRDataset(spr["dev"])
test_ds = SPRDataset(spr["test"])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# Model ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab,
        d_model=128,
        nhead=8,
        num_layers=2,
        num_classes=2,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        # mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1)
        return self.fc(x)


model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# Experiment data container ---------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ------------------------------------------------------------------
# Training loop ----------------------------------------------------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, preds, gts = 0.0, [], []
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
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    train_loss = total_loss / len(train_ds)
    train_f1 = f1_score(gts, preds, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)

    # Validation
    model.eval()
    val_loss, val_preds, val_gts = 0.0, [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_preds.extend(logits.argmax(-1).cpu().tolist())
            val_gts.extend(batch["labels"].cpu().tolist())
    val_loss /= len(dev_ds)
    val_f1 = f1_score(val_gts, val_preds, average="macro")
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}"
    )

# ------------------------------------------------------------------
# Test evaluation --------------------------------------------------
model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        test_preds.extend(logits.argmax(-1).cpu().tolist())
        test_gts.extend(batch["labels"].cpu().tolist())
test_f1 = f1_score(test_gts, test_preds, average="macro")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
print(f"Test macro_f1 = {test_f1:.4f}")

# ------------------------------------------------------------------
# Save experiment data & plot -------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure()
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["train"],
    label="train_loss",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["losses"]["val"],
    label="val_loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_SPR.png"))

plt.figure()
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"],
    label="train_f1",
)
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"],
    label="val_f1",
)
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "f1_curve_SPR.png"))
