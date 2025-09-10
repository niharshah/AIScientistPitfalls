import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from typing import List

# -------------------- paths / device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- data loading ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
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


def get_dataset() -> DatasetDict:
    possible = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for p in possible:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)

    # ---------- synthetic fallback ----------
    print("SPR_BENCH not found, building synthetic dataset")

    def build_rows(n):
        shapes = "ABCD"
        rows = []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def rows_to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"] = rows_to_ds(build_rows(2000))
    d["dev"] = rows_to_ds(build_rows(500))
    d["test"] = rows_to_ds(build_rows(500))
    return d


spr = get_dataset()

# -------------------- vocab ----------------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = min(100, max(map(len, spr["train"]["sequence"])))


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.y = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


batch_size = 128
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size)


# -------------------- model ----------------------------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# ------------------ experiment tracking ----------------
experiment_data = {"weight_decay": {"SPR_BENCH": {}}}  # will fill per weight_decay

weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
epochs = 5
best_val_f1, best_state, best_wd = -1, None, None


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            labels.extend(batch["label"].long().cpu().numpy())
    return f1_score(labels, preds, average="macro"), preds, labels


# ------------------- hyper-parameter sweep --------------
criterion = nn.BCEWithLogitsLoss()
for wd in weight_decays:
    print(f"\n=== Training with weight_decay = {wd} ===")
    # init containers
    exp_rec = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    experiment_data["weight_decay"]["SPR_BENCH"][str(wd)] = exp_rec
    # model / optim
    model = CharBiGRU(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    # epoch loop
    for ep in range(1, epochs + 1):
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            tr_labels.extend(batch["label"].long().cpu().numpy())
        train_f1 = f1_score(tr_labels, tr_preds, average="macro")
        val_f1, _, _ = evaluate(model, dev_loader)
        exp_rec["metrics"]["train_macro_f1"].append(train_f1)
        exp_rec["metrics"]["val_macro_f1"].append(val_f1)
        exp_rec["losses"]["train"].append(np.mean(tr_losses))
        exp_rec["losses"]["val"].append(0)  # will fill below
        exp_rec["epochs"].append(ep)
        # quick val loss for record
        with torch.no_grad():
            val_losses = []
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(
                    criterion(model(batch["input_ids"]), batch["label"]).item()
                )
        exp_rec["losses"]["val"][-1] = np.mean(val_losses)
        print(f"  Epoch {ep}: train_f1={train_f1:.4f}  val_f1={val_f1:.4f}")
    # test eval
    test_f1, t_preds, t_labels = evaluate(model, test_loader)
    exp_rec["predictions"] = t_preds
    exp_rec["ground_truth"] = t_labels
    print(f"  --> weight_decay {wd} Test macro-F1: {test_f1:.4f}")
    # track best on dev
    if exp_rec["metrics"]["val_macro_f1"][-1] > best_val_f1:
        best_val_f1 = exp_rec["metrics"]["val_macro_f1"][-1]
        best_state = model.state_dict()
        best_wd = wd

# ------------------- best model report ------------------
print(f"\nBest weight_decay based on dev set: {best_wd}  (val_f1={best_val_f1:.4f})")
best_model = CharBiGRU(vocab_size).to(device)
best_model.load_state_dict(best_state)
best_test_f1, _, _ = evaluate(best_model, test_loader)
print(f"Test Macro-F1 of best model: {best_test_f1:.4f}")

# ------------------- save data & plot -------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# (Optional) loss curve for best wd
rec = experiment_data["weight_decay"]["SPR_BENCH"][str(best_wd)]
plt.figure(figsize=(6, 4))
plt.plot(rec["epochs"], rec["losses"]["train"], label="train_loss")
plt.plot(rec["epochs"], rec["losses"]["val"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss curve wd={best_wd}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, f"loss_curve_wd_{best_wd}.png"))
plt.close()
