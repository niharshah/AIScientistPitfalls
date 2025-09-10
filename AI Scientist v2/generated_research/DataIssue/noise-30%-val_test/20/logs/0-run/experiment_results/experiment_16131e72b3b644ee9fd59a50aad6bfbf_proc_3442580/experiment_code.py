import os, pathlib, random, string, numpy as np, torch, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
# experiment dict
experiment_data = {
    "hidden_size": {"SPR_BENCH": {}}  # hyperparam_tuning_type_1  # dataset_name_1
}

# -------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------
# helper to load SPR benchmark (from prompt)
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


# -------------------------------------------------
# attempt to load dataset, otherwise create synthetic
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_data = DATA_PATH.exists()
if have_data:
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found, generating synthetic dataset.")

    def synth_split(n):
        rows = []
        for i in range(n):
            seq_len = random.randint(5, 15)
            seq = "".join(random.choices(string.ascii_uppercase[:10], k=seq_len))
            label = int(seq.count("A") % 2 == 0)  # simple parity rule
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    spr = DatasetDict()
    spr["train"] = load_dataset(
        "json", data_files={"train": synth_split(2000)}, split="train"
    )
    spr["dev"] = load_dataset(
        "json", data_files={"train": synth_split(400)}, split="train"
    )
    spr["test"] = load_dataset(
        "json", data_files={"train": synth_split(400)}, split="train"
    )

print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------
# build vocabulary
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq, max_len):
    ids = [vocab.get(ch, 1) for ch in seq][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


max_len = min(max(len(ex["sequence"]) for ex in spr["train"]), 120)


# -------------------------------------------------
# PyTorch Dataset wrapper
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(
                encode(ex["sequence"], max_len), dtype=torch.long
            ),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
        }


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# -------------------------------------------------
# Model definition
class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return self.fc(h.squeeze(0))


# -------------------------------------------------
hidden_sizes = [64, 128, 256, 512]
epochs = 5

for hs in hidden_sizes:
    print(f"\n=== Training with hidden_size = {hs} ===")
    # create entry in experiment_data
    experiment_data["hidden_size"]["SPR_BENCH"][hs] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = CharGRU(vocab_size, hidden=hs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # ------------------ train ------------------
        model.train()
        total_loss, total_items = 0.0, 0
        all_preds_train, all_labels_train = [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            total_items += batch["labels"].size(0)
            all_preds_train.extend(logits.argmax(1).cpu().numpy())
            all_labels_train.extend(batch["labels"].cpu().numpy())
        train_loss = total_loss / total_items
        train_f1 = f1_score(all_labels_train, all_preds_train, average="macro")

        # ------------------ validate ------------------
        model.eval()
        val_loss, val_items = 0.0, 0
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item() * batch["labels"].size(0)
                val_items += batch["labels"].size(0)
                all_preds_val.extend(logits.argmax(1).cpu().numpy())
                all_labels_val.extend(batch["labels"].cpu().numpy())
        val_loss /= val_items
        val_f1 = f1_score(all_labels_val, all_preds_val, average="macro")

        # store
        exp_entry = experiment_data["hidden_size"]["SPR_BENCH"][hs]
        exp_entry["losses"]["train"].append(train_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append(train_f1)
        exp_entry["metrics"]["val"].append(val_f1)

        print(
            f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} F1 {train_f1:.3f} "
            f"| ValLoss {val_loss:.4f} F1 {val_f1:.3f}"
        )

    # store final predictions / ground_truth
    exp_entry["predictions"] = all_preds_val
    exp_entry["ground_truth"] = all_labels_val

# -------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
