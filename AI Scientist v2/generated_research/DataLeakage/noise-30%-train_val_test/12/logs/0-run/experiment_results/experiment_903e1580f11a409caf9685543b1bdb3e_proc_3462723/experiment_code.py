# num_epochs_hparam_tuning.py
import os, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import List

# ---------------------- EXPERIMENT LOG -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "num_epochs": {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
            "best_epoch": None,
            "best_val_f1": 0.0,
        }
    }
}

# ---------------------- DEVICE ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------- DATASET --------------------------------------
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
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ---------------------- VOCAB ----------------------------------------
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(char_set)
stoi = {ch: i for i, ch in enumerate(itos)}
max_len = 128
num_classes = len(set(spr["train"]["label"]))


def encode(seq: str, max_len=128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    return ids + [stoi[PAD]] * (max_len - len(ids))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, max_len=128):
        self.data, self.max_len = hf_ds, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode(row["sequence"], self.max_len), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": torch.tensor(row["label"]),
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size)


# ---------------------- MODEL ----------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab, classes, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Linear(d_model, classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(x)


model = TinyTransformer(len(itos), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# ---------------------- TRAIN / EVAL ---------------------------------
def run(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------------- TRAIN LOOP with EARLY STOP -------------------
max_epochs, patience = 30, 5
stale = 0
for epoch in range(1, max_epochs + 1):
    tr_loss, tr_f1, _, _ = run(train_loader, True)
    val_loss, val_f1, val_pred, val_gt = run(dev_loader, False)

    log = experiment_data["num_epochs"]["SPR_BENCH"]
    log["epochs"].append(epoch)
    log["losses"]["train"].append(tr_loss)
    log["losses"]["val"].append(val_loss)
    log["metrics"]["train_f1"].append(tr_f1)
    log["metrics"]["val_f1"].append(val_f1)

    if val_f1 > log["best_val_f1"]:
        log["best_val_f1"], log["best_epoch"] = val_f1, epoch
        log["predictions"], log["ground_truth"] = val_pred, val_gt
        stale = 0
    else:
        stale += 1

    print(
        f"Epoch {epoch:02d}: "
        f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_macroF1={val_f1:.4f} (best={log['best_val_f1']:.4f})"
    )

    if stale >= patience:
        print("Early stopping triggered.")
        break

# ---------------------- SAVE RESULTS ---------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

plt.figure()
plt.plot(log["epochs"], log["losses"]["train"], label="train_loss")
plt.plot(log["epochs"], log["losses"]["val"], label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(log["epochs"], log["metrics"]["val_f1"], label="val_macro_f1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Validation Macro F1")
plt.savefig(os.path.join(working_dir, "f1_curve.png"))
plt.close()

print(f"Best Dev Macro_F1 = {log['best_val_f1']:.4f} @ epoch {log['best_epoch']}")
