import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------------- misc / seeds --------------------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# -------------------------- saving dir ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------- device --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------- data ----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    for split in ["train", "dev", "test"]:
        ds[split] = _load(f"{split}.csv")
    return ds


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)

    # synthetic fallback if real data missing
    print("SPR_BENCH not found, creating synthetic toy dataset")

    def synth(n):
        rows = []
        shapes = "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_hfds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    dset = DatasetDict()
    dset["train"] = to_hfds(synth(2000))
    dset["dev"] = to_hfds(synth(500))
    dset["test"] = to_hfds(synth(500))
    return dset


spr = get_dataset()

# -------------------------- vocab ---------------------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(list(set(all_text)))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
print("Vocab size:", vocab_size - 1)

max_len = min(100, max(len(s) for s in spr["train"]["sequence"]))


def encode(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.y = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.seq[idx]), dtype=torch.long)
        y = torch.tensor(int(self.y[idx]), dtype=torch.float)
        return {"input_ids": x, "label": y}


# -------------------------- model ---------------------------------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# -------------------- hyperparameter tuning -----------------------
embed_dims = [32, 64, 128, 256]
batch_size, lr, epochs = 128, 1e-3, 5

experiment_data = {"emb_dim": {}}
best_val_f1, best_state, best_dim = -1, None, None

for emb_dim in embed_dims:
    print(f"\n=== Training with emb_dim={emb_dim} ===")
    train_loader = DataLoader(
        SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)
    test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)

    model = CharBiGRU(vocab_size, emb_dim=emb_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logs = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_preds, tr_lbls = [], [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().numpy())
            tr_lbls.extend(batch["label"].cpu().int().numpy())
        train_f1 = f1_score(tr_lbls, tr_preds, average="macro")

        # ---- validation ----
        model.eval()
        val_loss, val_preds, val_lbls = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss.append(loss.item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().numpy())
                val_lbls.extend(batch["label"].cpu().int().numpy())
        val_f1 = f1_score(val_lbls, val_preds, average="macro")
        print(f"Epoch {ep}: val_loss={np.mean(val_loss):.4f}  val_F1={val_f1:.4f}")

        # ---- log ----
        logs["metrics"]["train_macro_f1"].append(train_f1)
        logs["metrics"]["val_macro_f1"].append(val_f1)
        logs["losses"]["train"].append(np.mean(tr_loss))
        logs["losses"]["val"].append(np.mean(val_loss))
        logs["epochs"].append(ep)

        # save best across epochs for this emb_dim
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            best_dim = emb_dim

    experiment_data["emb_dim"][emb_dim] = logs

print(f"\nBest emb_dim={best_dim} with dev F1={best_val_f1:.4f}")

# -------------------- test with best model ------------------------
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)

best_model = CharBiGRU(vocab_size, emb_dim=best_dim).to(device)
best_model.load_state_dict(best_state)
best_model.eval()

test_preds, test_lbls = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = best_model(batch["input_ids"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().numpy())
        test_lbls.extend(batch["label"].cpu().int().numpy())

test_f1 = f1_score(test_lbls, test_preds, average="macro")
print(f"Test Macro-F1 (best emb_dim={best_dim}): {test_f1:.4f}")

experiment_data["emb_dim"][best_dim]["predictions"] = test_preds
experiment_data["emb_dim"][best_dim]["ground_truth"] = test_lbls
experiment_data["emb_dim"][best_dim]["test_macro_f1"] = test_f1

# -------------------- save & plot -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot loss curves per emb_dim
plt.figure(figsize=(7, 4))
for emb_dim, logs in experiment_data["emb_dim"].items():
    plt.plot(logs["epochs"], logs["losses"]["val"], label=f"emb={emb_dim}")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.legend()
plt.title("Validation Loss vs Epochs (different emb_dim)")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "val_loss_emb_dim.png"))
plt.close()
