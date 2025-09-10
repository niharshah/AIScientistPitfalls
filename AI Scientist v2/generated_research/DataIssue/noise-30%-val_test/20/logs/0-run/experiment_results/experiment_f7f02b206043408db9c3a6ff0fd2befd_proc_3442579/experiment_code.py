import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------------------------------------
# experiment data container
experiment_data = {
    "learning_rate": {  # tuning type
        "SPR_BENCH": {}  # dataset name – per-lr sub-dicts will be inserted
    }
}

# ---------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------
# helper to load SPR benchmark
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


# ---------------------------------------------
# load dataset (fall back to synthetic)
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
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    spr = DatasetDict()
    for split, n in [("train", 2000), ("dev", 400), ("test", 400)]:
        spr[split] = load_dataset(
            "json", data_files={split: synth_split(n)}, split="train"
        )
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------
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
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


max_len = min(max(len(ex["sequence"]) for ex in spr["train"]), 120)


# ---------------------------------------------
# Torch Dataset wrapper
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


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------------------------------------------
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


# ---------------------------------------------
lrs_to_try = [3e-4, 5e-4, 1e-3, 3e-3]
epochs = 5

for lr in lrs_to_try:
    print(f"\n=== Training with learning rate {lr} ===")
    # set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # (re)initialise model & optimiser
    model = CharGRU(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create entry in experiment_data
    run_store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = run_store

    for epoch in range(1, epochs + 1):
        # -------- training ---------
        model.train()
        total_loss, total_items = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            n = batch["labels"].size(0)
            total_loss += loss.item() * n
            total_items += n
        train_loss = total_loss / total_items
        run_store["losses"]["train"].append(train_loss)

        # -------- validation --------
        model.eval()
        val_loss, val_items = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                n = batch["labels"].size(0)
                val_loss += loss.item() * n
                val_items += n
                preds = logits.argmax(dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        val_loss /= val_items
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["val"].append(macro_f1)
        print(
            f"LR {lr} | Epoch {epoch}: val_loss={val_loss:.4f}, Macro-F1={macro_f1:.4f}"
        )

    # store final epoch predictions
    run_store["predictions"] = all_preds
    run_store["ground_truth"] = all_labels

# ---------------------------------------------
# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
