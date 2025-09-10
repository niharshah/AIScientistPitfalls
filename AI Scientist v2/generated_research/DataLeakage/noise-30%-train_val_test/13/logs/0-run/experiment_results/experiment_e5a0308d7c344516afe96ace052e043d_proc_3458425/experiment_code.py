import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import string
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pathlib
from typing import List, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# 1. Data loading (real benchmark if present, synthetic otherwise)
# ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def create_synthetic_dataset(
    n_train=2000,
    n_dev=500,
    n_test=500,
    seq_len=12,
    vocab=list(string.ascii_uppercase),
    n_classes=3,
):
    def _gen(n):
        seqs, labels = [], []
        for _ in range(n):
            seqs.append("".join(random.choices(vocab, k=seq_len)))
            labels.append(random.randint(0, n_classes - 1))
        return Dataset.from_dict({"sequence": seqs, "label": labels})

    return DatasetDict(train=_gen(n_train), dev=_gen(n_dev), test=_gen(n_test))


try:
    SPR_PATH = pathlib.Path(os.getenv("SPR_PATH", "SPR_BENCH"))
    spr_data = load_spr_bench(SPR_PATH)
    dataset_name = "SPR_BENCH"
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load SPR_BENCH, falling back to synthetic data:", e)
    spr_data = create_synthetic_dataset()
    dataset_name = "synthetic"

# ------------------------------------------------------------------
# 2. Tokeniser
# ------------------------------------------------------------------
PAD_ID, UNK_ID = 0, 1


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {ch for s in seqs for ch in s}
    stoi = {ch: i + 2 for i, ch in enumerate(sorted(vocab))}
    stoi["<pad>"] = PAD_ID
    stoi["<unk>"] = UNK_ID
    return stoi


vocab = build_vocab(spr_data["train"]["sequence"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi.get(ch, UNK_ID) for ch in seq]


# Add encoded ids to dataset (map keeps lazy memory)
def encode_examples(example):
    example["input_ids"] = encode_sequence(example["sequence"], vocab)
    return example


spr_data = spr_data.map(encode_examples)

# Label mapping to contiguous ints
labels = sorted(set(spr_data["train"]["label"]))
label2id = {lbl: i for i, lbl in enumerate(labels)}
num_classes = len(labels)


def map_label(example):
    example["label_id"] = label2id[example["label"]]
    return example


spr_data = spr_data.map(map_label)


# ------------------------------------------------------------------
# 3. DataLoader with padding collate
# ------------------------------------------------------------------
def collate_fn(batch):
    input_lens = [len(x["input_ids"]) for x in batch]
    max_len = max(input_lens)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        ids = x["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [PAD_ID] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(x["label_id"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


train_loader = DataLoader(
    spr_data["train"], batch_size=64, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    spr_data["dev"], batch_size=128, shuffle=False, collate_fn=collate_fn
)


# ------------------------------------------------------------------
# 4. Model definition
# ------------------------------------------------------------------
class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        pos_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        # mean pooling excluding pads
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(x)
        return logits


model = SimpleTransformerClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=num_classes,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------------
experiment_data = {
    dataset_name: {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

n_epochs = 5
for epoch in range(1, n_epochs + 1):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    y_true_train, y_pred_train = [], []
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
        preds = logits.argmax(1).detach().cpu().numpy()
        y_pred_train.extend(preds)
        y_true_train.extend(batch["labels"].cpu().numpy())
    train_loss = running_loss / len(spr_data["train"])
    train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

    # ---- Validate ----
    model.eval()
    val_loss_total = 0.0
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            val_loss_total += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            y_pred_val.extend(preds)
            y_true_val.extend(batch["labels"].cpu().numpy())
    val_loss = val_loss_total / len(spr_data["dev"])
    val_f1 = f1_score(y_true_val, y_pred_val, average="macro")

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, train_f1={train_f1:.4f} | "
        f"validation_loss = {val_loss:.4f}, val_f1={val_f1:.4f}"
    )

    # log
    experiment_data[dataset_name]["epochs"].append(epoch)
    experiment_data[dataset_name]["losses"]["train"].append(train_loss)
    experiment_data[dataset_name]["losses"]["val"].append(val_loss)
    experiment_data[dataset_name]["metrics"]["train_macro_f1"].append(train_f1)
    experiment_data[dataset_name]["metrics"]["val_macro_f1"].append(val_f1)

# Save predictions of last validation
experiment_data[dataset_name]["predictions"] = y_pred_val
experiment_data[dataset_name]["ground_truth"] = y_true_val

# ------------------------------------------------------------------
# 6. Save metrics and plots
# ------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Plot
plt.figure()
plt.plot(
    experiment_data[dataset_name]["epochs"],
    experiment_data[dataset_name]["losses"]["train"],
    label="train_loss",
)
plt.plot(
    experiment_data[dataset_name]["epochs"],
    experiment_data[dataset_name]["losses"]["val"],
    label="val_loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))

plt.figure()
plt.plot(
    experiment_data[dataset_name]["epochs"],
    experiment_data[dataset_name]["metrics"]["train_macro_f1"],
    label="train_macro_f1",
)
plt.plot(
    experiment_data[dataset_name]["epochs"],
    experiment_data[dataset_name]["metrics"]["val_macro_f1"],
    label="val_macro_f1",
)
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.legend()
plt.savefig(os.path.join(working_dir, f"{dataset_name}_f1_curve.png"))
print("Training complete. Metrics and plots saved in 'working/' directory.")
