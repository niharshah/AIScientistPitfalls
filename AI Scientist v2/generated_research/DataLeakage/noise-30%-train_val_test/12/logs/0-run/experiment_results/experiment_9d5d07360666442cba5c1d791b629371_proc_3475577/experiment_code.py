# No-Padding-Mask Ablation Study  (single-file, self-contained)
import os, pathlib, numpy as np, torch, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
from typing import List, Dict

# ---------- HOUSE-KEEPING ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- DATA ----------
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
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ---------- VOCAB ----------
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}
max_len = 128
num_classes = len(set(spr["train"]["label"]))


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


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
        attention_mask = (input_ids != stoi[PAD]).long()  # still produced but unused
        label = torch.tensor(row["label"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# ---------- MODEL  (No-Padding-Mask) ----------
class TinyTransformerNoPad(nn.Module):
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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):  # attention_mask ignored
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        x = self.encoder(x)  # NO src_key_padding_mask
        x = x.mean(1)  # mean over ALL tokens
        return self.fc(x)


# ---------- TRAIN / EVAL LOOP ----------
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
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ---------- EXPERIMENT ----------
experiment_data: Dict = {"no_padding_mask_ablation": {}}
batch_sizes = [32, 64, 128, 256]
epochs = 5
criterion = nn.CrossEntropyLoss()

for bs in batch_sizes:
    print(f"\n=== No-Pad-Mask Ablation: batch_size={bs} ===")
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"], max_len), batch_size=bs, shuffle=True
    )
    dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=bs)
    model = TinyTransformerNoPad(len(itos), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    run_stats = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_preds, val_gts = run_loader(model, dev_loader, criterion)

        run_stats["epochs"].append(epoch)
        run_stats["losses"]["train"].append(tr_loss)
        run_stats["losses"]["val"].append(val_loss)
        run_stats["metrics"]["train_f1"].append(tr_f1)
        run_stats["metrics"]["val_f1"].append(val_f1)
        if epoch == epochs:
            run_stats["predictions"] = val_preds
            run_stats["ground_truth"] = val_gts
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_macroF1={val_f1:.4f}"
        )

    # plots
    plt.figure()
    plt.plot(run_stats["epochs"], run_stats["losses"]["train"], label="train_loss")
    plt.plot(run_stats["epochs"], run_stats["losses"]["val"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"No-Pad Ablation Loss (bs={bs})")
    plt.savefig(os.path.join(working_dir, f"loss_curve_nopad_bs{bs}.png"))
    plt.close()

    plt.figure()
    plt.plot(run_stats["epochs"], run_stats["metrics"]["val_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"No-Pad Ablation F1 (bs={bs})")
    plt.savefig(os.path.join(working_dir, f"f1_curve_nopad_bs{bs}.png"))
    plt.close()

    experiment_data["no_padding_mask_ablation"][bs] = run_stats

# ---------- SAVE ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nAblation complete. Data saved to working/experiment_data.npy")
