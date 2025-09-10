import os, pathlib, random, string, time, json, math

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

# --------------------------------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------------------
# 1) Load or create SPR benchmark -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def build_synthetic(
    n_train=2000, n_dev=500, n_test=1000, seq_len=15, vocab=list(string.ascii_lowercase)
):
    def make_split(n):
        seqs, labels = [], []
        for _ in range(n):
            label = random.randint(0, 3)
            # very naive rule: label determines parity of vowels in string
            s = "".join(random.choice(vocab) for _ in range(seq_len))
            seqs.append(s)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    data = {
        "train": make_split(n_train),
        "dev": make_split(n_dev),
        "test": make_split(n_test),
    }
    return DatasetDict({k: HFDataset.from_dict(v) for k, v in data.items()})


SPR_PATH = pathlib.Path(os.getcwd()) / "SPR_BENCH"
if SPR_PATH.exists():
    print("Loading real SPR_BENCH from", SPR_PATH)
    dsets = load_spr_bench(SPR_PATH)
else:
    print("SPR_BENCH not found – building synthetic data")
    dsets = build_synthetic()

# --------------------------------------------------------------------------------
# 2) Vocabulary and encoding ------------------------------------------------------
PAD, UNK = 0, 1


def build_vocab(dataset):
    vocab = {c for seq in dataset["sequence"] for c in seq}
    idx = {ch: i + 2 for i, ch in enumerate(sorted(vocab))}
    idx["<PAD>"] = PAD
    idx["<UNK>"] = UNK
    return idx


vocab2idx = build_vocab(dsets["train"])
idx2vocab = {i: ch for ch, i in vocab2idx.items()}
vocab_size = len(vocab2idx)
print("Vocab size:", vocab_size)

max_len = max(len(seq) for seq in dsets["train"]["sequence"])
print("Max sequence length:", max_len)


def encode(seq, max_len=max_len):
    ids = [vocab2idx.get(c, UNK) for c in seq[:max_len]]
    if len(ids) < max_len:
        ids.extend([PAD] * (max_len - len(ids)))
    return ids


for split in dsets:
    dsets[split] = dsets[split].map(lambda x: {"input_ids": encode(x["sequence"])})


# --------------------------------------------------------------------------------
# 3) Torch dataset ----------------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


batch_size = 64
train_loader = DataLoader(
    SPRTorchDataset(dsets["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(dsets["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(dsets["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)

num_classes = len(set(dsets["train"]["label"]))
print("Num classes:", num_classes)


# --------------------------------------------------------------------------------
# 4) Model ------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=4, max_len=128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):  # x: [B, L]
        emb = self.embed(x) + self.pos_embed[:, : x.size(1), :]
        enc = self.encoder(emb)  # [B, L, D]
        pooled = enc.mean(dim=1)  # simple mean pooling
        return self.fc(pooled)


model = TransformerClassifier(
    vocab_size=vocab_size, num_classes=num_classes, max_len=max_len
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------------------------------------------------------------------------
# 5) Training ---------------------------------------------------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.set_grad_enabled(train):
            outputs = model(batch["input_ids"])
            loss = criterion(outputs, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["input_ids"].size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        labs = batch["labels"].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labs)
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

n_epochs = 10
for epoch in range(1, n_epochs + 1):
    train_loss, train_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, train=False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MacroF1={val_f1:.4f}"
    )

# --------------------------------------------------------------------------------
# 6) Final test evaluation --------------------------------------------------------
test_loss, test_f1, test_preds, test_labels = run_epoch(test_loader, train=False)
print(f"\nTest MacroF1 = {test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --------------------------------------------------------------------------------
# 7) Plot learning curves ---------------------------------------------------------
import matplotlib.pyplot as plt

epochs = range(1, n_epochs + 1)
plt.figure()
plt.plot(epochs, experiment_data["SPR_BENCH"]["losses"]["train"], label="train_loss")
plt.plot(epochs, experiment_data["SPR_BENCH"]["losses"]["val"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(epochs, experiment_data["SPR_BENCH"]["metrics"]["train"], label="train_F1")
plt.plot(epochs, experiment_data["SPR_BENCH"]["metrics"]["val"], label="val_F1")
plt.xlabel("Epoch")
plt.ylabel("MacroF1")
plt.title("MacroF1 curves")
plt.legend()
plt.savefig(os.path.join(working_dir, "f1_curve.png"))
plt.close()
