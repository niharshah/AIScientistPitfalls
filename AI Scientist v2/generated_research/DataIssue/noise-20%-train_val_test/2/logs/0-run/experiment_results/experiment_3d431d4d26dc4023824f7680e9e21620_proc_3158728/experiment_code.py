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

import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
experiment_data = {"num_epochs": {}}  # hyper-parameter tuning type


# -------------------- dataset loader utility --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for s in ["train", "dev", "test"]:
        dset[s if s != "dev" else "dev"] = _load(f"{s}.csv")
    return dset


# -------------------- PyTorch dataset --------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.pad_id, self.max_len = vocab, vocab["<pad>"], max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq, label = self.seqs[idx], self.labels[idx]
        ids = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in seq[: self.max_len]]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -------------------- model --------------------
class SPRModel(nn.Module):
    def __init__(
        self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, max_len=128
    ):
        super().__init__()
        self.embed, self.pos = nn.Embedding(vocab_size, d_model), nn.Parameter(
            torch.randn(1, max_len, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 256)
        self.transformer, self.cls = nn.TransformerEncoder(
            enc_layer, num_layers
        ), nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, : x.size(1)]
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1).mean(1)
        return self.cls(x)


# -------------------- training utils --------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    tot_loss, pred, true = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        pred.extend(out.argmax(1).cpu().numpy())
        true.extend(batch["labels"].cpu().numpy())
    return tot_loss / len(loader.dataset), f1_score(true, pred, average="macro")


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tot_loss, pred, true = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        tot_loss += criterion(out, batch["labels"]).item() * batch["labels"].size(0)
        pred.extend(out.argmax(1).cpu().numpy())
        true.extend(batch["labels"].cpu().numpy())
    return (
        tot_loss / len(loader.dataset),
        f1_score(true, pred, average="macro"),
        pred,
        true,
    )


# -------------------- main routine --------------------
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    spr = load_spr_bench(DATA_PATH)

    # vocab & datasets ---------------------------------------------------------
    chars = set("".join(spr["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({ch: i + 2 for i, ch in enumerate(sorted(chars))})
    max_len = min(128, max(len(s) for s in spr["train"]["sequence"]))

    train_loader = DataLoader(
        SPRTorchDataset(spr["train"], vocab, max_len), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(SPRTorchDataset(spr["dev"], vocab, max_len), batch_size=256)
    test_loader = DataLoader(
        SPRTorchDataset(spr["test"], vocab, max_len), batch_size=256
    )

    num_classes = len(set(spr["train"]["label"]))
    criterion = nn.CrossEntropyLoss()

    epoch_grid = [5, 10, 20, 30]  # hyper-parameter values
    patience = 5

    for max_epochs in epoch_grid:
        key = f"epochs_{max_epochs}"
        experiment_data["num_epochs"][key] = {
            "metrics": {"train_macro_f1": [], "val_macro_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }

        model = SPRModel(len(vocab), num_classes, max_len=max_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val, wait, best_state = 0.0, 0, None
        for epoch in range(1, max_epochs + 1):
            tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
            print(
                f"[{key}] Epoch {epoch}: val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}"
            )

            # store
            ed = experiment_data["num_epochs"][key]
            ed["epochs"].append(epoch)
            ed["losses"]["train"].append(tr_loss)
            ed["losses"]["val"].append(val_loss)
            ed["metrics"]["train_macro_f1"].append(tr_f1)
            ed["metrics"]["val_macro_f1"].append(val_f1)

            # early stopping
            if val_f1 > best_val:
                best_val, wait, best_state = val_f1, 0, model.state_dict()
            else:
                wait += 1
            if wait >= patience:
                print(
                    f"Early stopping at epoch {epoch} (best val_macro_f1={best_val:.4f})"
                )
                break

        # load best model and evaluate on test set
        if best_state is not None:
            model.load_state_dict(best_state)
        test_loss, test_f1, preds, gts = eval_epoch(model, test_loader, criterion)
        print(f"[{key}] Test macro_f1={test_f1:.4f}")
        ed = experiment_data["num_epochs"][key]
        ed["predictions"], ed["ground_truth"] = preds, gts
        ed["test_macro_f1"], ed["test_loss"] = test_f1, test_loss

    # -------------------- save all experiment data --------------------
    working_dir = os.path.join(os.getcwd(), "working")
    os.makedirs(working_dir, exist_ok=True)
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Execute immediately
main()
