import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from typing import List

# ------------- folders & meta -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"hidden_size": {}}  # container required by guideline

# ------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------- data loading --------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    for split in ["train", "dev", "test"]:
        ds[split] = _load(f"{split}.csv")
    return ds


def get_dataset() -> DatasetDict:
    # search local real dataset
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)
    # otherwise create synthetic toy dataset
    print("SPR_BENCH not found, creating synthetic toy dataset")

    def synth(n):
        rows, shapes = [], "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        to_ds(synth(2000)),
        to_ds(synth(500)),
        to_ds(synth(500)),
    )
    return dset


spr = get_dataset()

# ------------- vocab ----------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
print("Vocab size:", vocab_size - 1)
max_len = min(100, max(len(s) for s in spr["train"]["sequence"]))


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq, self.y = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


batch_size = 128
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)


# ------------- model ----------------------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# ------------- hyper-parameter sweep ------
hidden_sizes = [64, 128, 256, 512]
num_epochs = 5
best_val_f1, best_state, best_hid = 0.0, None, None

for hid in hidden_sizes:
    print(f"\n=== Training with hidden_size={hid} ===")
    # containers for this run
    exp_run = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
    }
    # model, optimiser
    model = CharBiGRU(vocab_size, emb_dim=64, hid=hid).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_pred, tr_gold = [], [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["label"])
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())
            tr_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            tr_gold.extend(batch["label"].long().cpu().numpy())
        train_f1 = f1_score(tr_gold, tr_pred, average="macro")

        # ---- validate ----
        model.eval()
        val_loss, val_pred, val_gold = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                val_loss.append(crit(logits, batch["label"]).item())
                val_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
                val_gold.extend(batch["label"].long().cpu().numpy())
        val_f1 = f1_score(val_gold, val_pred, average="macro")

        print(
            f"  Epoch {epoch}: val_loss={np.mean(val_loss):.4f} val_macro_f1={val_f1:.4f}"
        )

        # store stats
        exp_run["metrics"]["train_macro_f1"].append(train_f1)
        exp_run["metrics"]["val_macro_f1"].append(val_f1)
        exp_run["losses"]["train"].append(np.mean(tr_loss))
        exp_run["losses"]["val"].append(np.mean(val_loss))
        exp_run["epochs"].append(epoch)

    # save run info
    experiment_data["hidden_size"][str(hid)] = exp_run

    # keep best
    if exp_run["metrics"]["val_macro_f1"][-1] > best_val_f1:
        best_val_f1 = exp_run["metrics"]["val_macro_f1"][-1]
        best_state = model.state_dict()
        best_hid = hid

print(f"\nBest hidden_size={best_hid} with dev macro-F1={best_val_f1:.4f}")

# ------------- test evaluation with best ---
best_model = CharBiGRU(vocab_size, emb_dim=64, hid=best_hid).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_pred, test_gold = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        test_pred.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
        test_gold.extend(batch["label"].long().cpu().numpy())
test_macro_f1 = f1_score(test_gold, test_pred, average="macro")
print(f"Test Macro-F1 (best model): {test_macro_f1:.4f}")

experiment_data["hidden_size"]["best_model"] = {
    "hidden_size": best_hid,
    "test_macro_f1": test_macro_f1,
    "predictions": np.array(test_pred),
    "ground_truth": np.array(test_gold),
}

# ------------- save & plot -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# loss curves for each hid
plt.figure(figsize=(8, 5))
for hid in hidden_sizes:
    plt.plot(
        experiment_data["hidden_size"][str(hid)]["epochs"],
        experiment_data["hidden_size"][str(hid)]["losses"]["val"],
        label=f"hid{hid}",
    )
plt.xlabel("Epoch")
plt.ylabel("Val loss")
plt.title("Val loss per hidden size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "val_loss_hidden_size.png"))
plt.close()
