import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
from typing import List

# -------------------------------------------------------------------- EXPERIMENT LOG
experiment_data = {
    "n_layers": {
        "SPR_BENCH": {
            "depths": [],
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],  # list per depth (last epoch dev preds)
            "ground_truth": [],  # list per depth
            "epoch_curves": {},  # depth -> {'train_f1':[],'val_f1':[],'train_loss':[],'val_loss':[]}
        }
    }
}

# -------------------------------------------------------------------- DEVICE & WORKDIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print("Using device:", device)


# -------------------------------------------------------------------- DATA
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded dataset:", {k: len(v) for k, v in spr.items()})

# ------------------------------ VOCAB & ENCODER
PAD, UNK = "<pad>", "<unk>"
char_set = set()
for ex in spr["train"]:
    char_set.update(list(ex["sequence"]))
itos = [PAD, UNK] + sorted(char_set)
stoi = {ch: i for i, ch in enumerate(itos)}


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


max_len = 128
num_classes = len(set(spr["train"]["label"]))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data = hf_dataset
        self.maxlen = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode(row["sequence"], self.maxlen), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": label}


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
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask):
        x = self.embed(ids) + self.pos[:, : ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.fc(x)


# -------------------------------------------------------------------- TRAIN / EVAL
def run_loader(model, loader, criterion, optimizer=None):
    train = optimizer is not None
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
            preds.extend(out.argmax(-1).detach().cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# -------------------------------------------------------------------- SWEEP
depth_values = [1, 2, 4, 6]
epochs = 5

for depth in depth_values:
    print(f"\n=== Training TinyTransformer with n_layers={depth} ===")
    model = TinyTransformer(len(itos), num_classes, n_layers=depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    curve = {"train_f1": [], "val_f1": [], "train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(model, train_loader, criterion, optimizer)
        vl_loss, vl_f1, vl_preds, vl_gts = run_loader(model, dev_loader, criterion)

        curve["train_loss"].append(tr_loss)
        curve["val_loss"].append(vl_loss)
        curve["train_f1"].append(tr_f1)
        curve["val_f1"].append(vl_f1)

        print(
            f"Depth {depth} Epoch {ep}: train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} val_F1={vl_f1:.4f}"
        )

    # ---- log final epoch metrics
    exp_ds = experiment_data["n_layers"]["SPR_BENCH"]
    exp_ds["depths"].append(depth)
    exp_ds["metrics"]["train_f1"].append(curve["train_f1"][-1])
    exp_ds["metrics"]["val_f1"].append(curve["val_f1"][-1])
    exp_ds["losses"]["train"].append(curve["train_loss"][-1])
    exp_ds["losses"]["val"].append(curve["val_loss"][-1])
    exp_ds["predictions"].append(vl_preds)
    exp_ds["ground_truth"].append(vl_gts)
    exp_ds["epoch_curves"][depth] = curve

    # ---- plotting per depth
    plt.figure()
    plt.plot(range(1, epochs + 1), curve["train_loss"], label="train_loss")
    plt.plot(range(1, epochs + 1), curve["val_loss"], label="val_loss")
    plt.title(f"Loss (n_layers={depth})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_depth_{depth}.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), curve["val_f1"], label="val_f1")
    plt.title(f"Val Macro F1 (n_layers={depth})")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"f1_depth_{depth}.png"))
    plt.close()

    # free gpu
    del model
    torch.cuda.empty_cache()

# -------------------------------------------------------------------- SAVE RESULTS
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
best_idx = int(np.argmax(experiment_data["n_layers"]["SPR_BENCH"]["metrics"]["val_f1"]))
best_depth = experiment_data["n_layers"]["SPR_BENCH"]["depths"][best_idx]
best_f1 = experiment_data["n_layers"]["SPR_BENCH"]["metrics"]["val_f1"][best_idx]
print(f"\nBest depth={best_depth} with Dev Macro F1={best_f1:.4f}")
