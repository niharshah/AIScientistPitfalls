import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from typing import Dict, List
from datasets import DatasetDict, load_dataset

# ------------------------------------------------------------------
# 0. Repro / env ----------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 1. Data loading ---------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", spr)


# ------------------------------------------------------------------
# 2. Vocab + encoding ----------------------------------------------
def build_vocab(dsets) -> Dict[str, int]:
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    return {ch: i + 1 for i, ch in enumerate(sorted(chars))}  # 0 reserved for PAD


vocab = build_vocab(spr)
vocab_size = len(vocab) + 1
max_len = max(max(len(s) for s in split["sequence"]) for split in spr.values())
print(f"Vocab size: {vocab_size}  |  Max len: {max_len}")


def encode_sequence(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[ch] for ch in seq]


def pad(seq_ids: List[int], max_len: int) -> List[int]:
    return (seq_ids + [0] * max_len)[:max_len]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len):
        self.labels = hf_split["label"]
        self.seqs = hf_split["sequence"]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode_sequence(self.seqs[idx], self.vocab), self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


train_ds = SPRTorchDataset(spr["train"], vocab, max_len)
dev_ds = SPRTorchDataset(spr["dev"], vocab, max_len)
test_ds = SPRTorchDataset(spr["test"], vocab, max_len)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# ------------------------------------------------------------------
# 3. Model ----------------------------------------------------------
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)  # h shape: [2,B,H]
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# ------------------------------------------------------------------
# 4. Hyper-param sweep ---------------------------------------------
learning_rates = [5e-4, 1e-3, 2e-3]
epochs = 5

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"learning_rate": {}}

for lr in learning_rates:
    lr_key = f"lr_{lr:.0e}"
    print(f"\n=== Training with learning rate {lr} ===")
    # prepare containers
    experiment_data["learning_rate"][lr_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # model, criterion, optimizer
    model = GRUBaseline(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -- training epochs ------------------------------------------
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["labels"].size(0)
        train_loss = running_loss / len(train_ds)
        # train MCC quick pass
        with torch.no_grad():
            preds, labels = [], []
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                p = model(batch["input_ids"]).sigmoid().cpu().numpy() > 0.5
                preds.append(p)
                labels.append(batch["labels"].cpu().numpy())
            train_mcc = matthews_corrcoef(np.concatenate(labels), np.concatenate(preds))
        # validation
        model.eval()
        val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item() * batch["labels"].size(0)
                preds.append((logits.sigmoid() > 0.5).cpu().numpy())
                labels.append(batch["labels"].cpu().numpy())
        val_loss /= len(dev_ds)
        val_mcc = matthews_corrcoef(np.concatenate(labels), np.concatenate(preds))
        # store
        ed = experiment_data["learning_rate"][lr_key]
        ed["losses"]["train"].append(train_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(train_mcc)
        ed["metrics"]["val"].append(val_mcc)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_MCC={val_mcc:.4f}"
        )

    # -- test evaluation ------------------------------------------
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    test_mcc = matthews_corrcoef(labels, preds)
    print(f"Test MCC @ {lr}: {test_mcc:.4f}")
    ed = experiment_data["learning_rate"][lr_key]
    ed["predictions"] = preds
    ed["ground_truth"] = labels
    # free memory
    del model
    torch.cuda.empty_cache()

# ------------------------------------------------------------------
# 5. Save -----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
