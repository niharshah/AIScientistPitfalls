# ------------------------------------------------------------
#  TinyTransformer – learning-rate hyper-parameter search
# ------------------------------------------------------------
import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict
from sklearn.metrics import f1_score
from typing import List

# -------------------------------------------------------------------- EXPERIMENT DATA STRUCTURE
experiment_data = {
    "learning_rate": {  #  <-- hyper-param tuning type
        "SPR_BENCH": {  #  <-- dataset name
            "lr_values": [],
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs_record": [],  # per-run epoch indices
        }
    }
}

# -------------------------------------------------------------------- SAVING / PLOT DIR
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------- DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------------------------- DATA
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
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
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
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids[:max_len]


max_len = 128
num_classes = len(set(spr["train"]["label"]))


# -------------------------------------------------------------------- DATASET WRAPPER
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data, self.max_len = hf_dataset, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode(row["sequence"], self.max_len), dtype=torch.long)
        attn = (ids != stoi[PAD]).long()
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=batch_size)


# -------------------------------------------------------------------- MODEL
class TinyTransformer(nn.Module):
    def __init__(self, vocab, classes, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(x)


# -------------------------------------------------------------------- TRAIN / EVAL
criterion = nn.CrossEntropyLoss()


def run_loader(model, loader, train_flag, optimizer=None):
    model.train() if train_flag else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_flag):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).detach().cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


# -------------------------------------------------------------------- LEARNING-RATE GRID SEARCH
lr_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
epochs = 5

for lr in lr_grid:
    print(f"\n=== Training with learning rate {lr} ===")
    # model & optimizer fresh each run
    model = TinyTransformer(len(itos), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # run-level logging
    run_train_losses, run_val_losses = [], []
    run_train_f1s, run_val_f1s = [], []
    run_epochs = []

    for ep in range(1, epochs + 1):
        train_loss, train_f1, _, _ = run_loader(model, train_loader, True, optimizer)
        val_loss, val_f1, val_pred, gt = run_loader(model, dev_loader, False)

        run_epochs.append(ep)
        run_train_losses.append(train_loss)
        run_val_losses.append(val_loss)
        run_train_f1s.append(train_f1)
        run_val_f1s.append(val_f1)

        print(
            f"  Epoch {ep}: train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_F1 {val_f1:.4f}"
        )

    # -------------- store results for this LR -----------------
    experiment_data["learning_rate"]["SPR_BENCH"]["lr_values"].append(lr)
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["train"].append(
        run_train_losses
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["losses"]["val"].append(
        run_val_losses
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["train_f1"].append(
        run_train_f1s
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val_f1"].append(
        run_val_f1s
    )
    experiment_data["learning_rate"]["SPR_BENCH"]["predictions"].append(val_pred)
    experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"].append(gt)
    experiment_data["learning_rate"]["SPR_BENCH"]["epochs_record"].append(run_epochs)

    # plot for this LR
    plt.figure()
    plt.plot(run_epochs, run_train_losses, label="train_loss")
    plt.plot(run_epochs, run_val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LR {lr} Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_lr_{lr}.png"))
    plt.close()

    plt.figure()
    plt.plot(run_epochs, run_val_f1s, label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"LR {lr} Val F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"f1_curve_lr_{lr}.png"))
    plt.close()

# -------------------------------------------------------------------- SAVE
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# print best LR summary
best_idx = int(
    np.argmax(
        [
            runs[-1]
            for runs in experiment_data["learning_rate"]["SPR_BENCH"]["metrics"][
                "val_f1"
            ]
        ]
    )
)
best_lr = experiment_data["learning_rate"]["SPR_BENCH"]["lr_values"][best_idx]
best_f1 = experiment_data["learning_rate"]["SPR_BENCH"]["metrics"]["val_f1"][best_idx][
    -1
]
print(f"\nBest LR {best_lr} achieved final dev Macro-F1 {best_f1:.4f}")
