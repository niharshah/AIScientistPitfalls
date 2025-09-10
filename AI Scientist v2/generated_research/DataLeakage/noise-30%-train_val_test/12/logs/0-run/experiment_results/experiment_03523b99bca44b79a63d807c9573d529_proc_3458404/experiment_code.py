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

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pathlib
from typing import List, Dict

# Device handling -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------- DATA LOAD
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset loaded:", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------- VOCAB
PAD, UNK = "<pad>", "<unk>"
char_set = set()
for ex in spr["train"]:
    char_set.update(list(ex["sequence"]))
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


max_len = 128
num_classes = len(set(spr["train"]["label"]))


# -------------------------------------------------------------------- DATASET WRAPPER
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data = hf_dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_ids = torch.tensor(
            encode(row["sequence"], self.max_len), dtype=torch.long
        )
        attention_mask = (input_ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=batch_size)


# -------------------------------------------------------------------- MODEL
class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )  # mean pool
        return self.fc(x)


model = TinyTransformer(len(itos), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# -------------------------------------------------------------------- EXPERIMENT DATA
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------------------------------------------------------- TRAIN / EVAL FUNCTIONS
def run_loader(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    preds, gts = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(outputs.argmax(-1).detach().cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------------------------------------------------------- TRAIN LOOP
epochs = 5
for epoch in range(1, epochs + 1):
    train_loss, train_f1, _, _ = run_loader(train_loader, train=True)
    val_loss, val_f1, val_preds, val_gts = run_loader(dev_loader, train=False)

    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    if epoch == epochs:
        experiment_data["SPR_BENCH"]["predictions"] = val_preds
        experiment_data["SPR_BENCH"]["ground_truth"] = val_gts

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macroF1={val_f1:.4f}"
    )

# -------------------------------------------------------------------- SAVE METRICS & PLOTS
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

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
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(
    experiment_data["SPR_BENCH"]["epochs"],
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"],
    label="val_macro_f1",
)
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Validation Macro F1")
plt.savefig(os.path.join(working_dir, "f1_curve.png"))
plt.close()

print(
    f'Final Dev Macro_F1: {experiment_data["SPR_BENCH"]["metrics"]["val_f1"][-1]:.4f}'
)
