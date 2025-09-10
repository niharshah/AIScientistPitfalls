import os, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from typing import List

# --------------------------- house-keeping ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- data ------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset loaded:", {k: len(v) for k, v in spr.items()})

PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(char_set)
stoi = {ch: i for i, ch in enumerate(itos)}


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


max_len, num_classes = 128, len(set(spr["train"]["label"]))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data, self.max_len = hf_dataset, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_ids = torch.tensor(encode(row["sequence"], self.max_len))
        attention_mask = (input_ids != stoi[PAD]).long()
        label = torch.tensor(row["label"])
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


# --------------------------- model -----------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask):
        x = self.embed(ids) + self.pos_embed[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.fc(x)


criterion = nn.CrossEntropyLoss()


def run_loader(model, loader, train=False, optimizer=None):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(outputs, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(outputs.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


# --------------------------- hyper-param sweep -----------------------------
weight_decays = [0.0, 1e-4, 1e-3, 5e-3]
epochs = 5

experiment_data = {"weight_decay": {}}

for wd in weight_decays:
    print(f"\n===== training with weight_decay={wd} =====")
    model = TinyTransformer(len(itos), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=wd)

    run_store = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(
            model, train_loader, train=True, optimizer=optimizer
        )
        val_loss, val_f1, val_preds, val_gts = run_loader(model, dev_loader)
        run_store["epochs"].append(ep)
        run_store["losses"]["train"].append(tr_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["train_f1"].append(tr_f1)
        run_store["metrics"]["val_f1"].append(val_f1)
        if ep == epochs:
            run_store["predictions"] = val_preds
            run_store["ground_truth"] = val_gts
        print(
            f"Ep {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f}"
        )

    # ------------- plot for this hyper-param -----------------
    plt.figure()
    plt.plot(run_store["epochs"], run_store["losses"]["train"], label="train")
    plt.plot(run_store["epochs"], run_store["losses"]["val"], label="val")
    plt.title(f"Loss (wd={wd})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_wd{wd}.png"))
    plt.close()

    plt.figure()
    plt.plot(run_store["epochs"], run_store["metrics"]["val_f1"])
    plt.title(f"Val F1 (wd={wd})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.savefig(os.path.join(working_dir, f"f1_wd{wd}.png"))
    plt.close()

    experiment_data["weight_decay"][wd] = {"SPR_BENCH": run_store}

# --------------------------- save everything -------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nFinished hyper-parameter tuning. Results saved to 'experiment_data.npy'.")
