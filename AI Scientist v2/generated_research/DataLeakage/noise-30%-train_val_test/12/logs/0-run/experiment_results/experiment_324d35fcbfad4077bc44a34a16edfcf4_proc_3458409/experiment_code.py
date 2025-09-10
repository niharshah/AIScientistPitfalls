import os, random, string, time
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
import matplotlib.pyplot as plt
import pathlib

# ------------------------------------------------------------------
# working dir + experiment store
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR": {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
# ------------------------------------------------------------------
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# 1. DATA --------------------------------------------------------------------
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


def make_synthetic(split_size):
    sequences = []
    labels = []
    vocab = list(string.ascii_uppercase[:6])  # 6 symbols
    for _ in range(split_size):
        length = random.randint(5, 20)
        seq = "".join(random.choices(vocab, k=length))
        lbl = random.randint(0, 4)  # 5 classes
        sequences.append(seq)
        labels.append(lbl)
    return {"id": list(range(split_size)), "sequence": sequences, "label": labels}


# try loading real data, else fabricate synthetic
data_path = pathlib.Path(os.environ.get("SPR_DATA_PATH", "SPR_BENCH"))
if data_path.exists() and (data_path / "train.csv").exists():
    print("Loading SPR_BENCH from disk …")
    dsets = load_spr_bench(data_path)
else:
    print("SPR_BENCH not found – generating synthetic data …")
    dsets = DatasetDict()
    dsets["train"] = HFDataset.from_dict(make_synthetic(4000))
    dsets["dev"] = HFDataset.from_dict(make_synthetic(800))
    dsets["test"] = HFDataset.from_dict(make_synthetic(800))

# ------------------------------------------------------------------
# 2. TOKENISER + ENCODER ------------------------------------------------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(dataset):
    counter = Counter()
    for seq in dataset["sequence"]:
        counter.update(list(seq.strip()))
    vocab = [PAD, UNK] + sorted(counter.keys())
    stoi = {s: i for i, s in enumerate(vocab)}
    return vocab, stoi


vocab, stoi = build_vocab(dsets["train"])
vocab_size = len(vocab)
print(f"Vocab size = {vocab_size}")


def encode(seq):
    return [stoi.get(ch, stoi[UNK]) for ch in list(seq.strip())]


for split in ["train", "dev", "test"]:
    dsets[split] = dsets[split].map(lambda ex: {"input_ids": encode(ex["sequence"])})


# ------------------------------------------------------------------
# 3. PYTORCH DATASET ----------------------------------------------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(x["input_ids"]) for x in batch]
    max_len = max(lengths)
    input_ids = []
    attention = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad_len,), stoi[PAD])]))
        attention.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))
        labels.append(item["labels"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention),
        "labels": torch.stack(labels),
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorchSet(dsets["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRTorchSet(dsets["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchSet(dsets["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ------------------------------------------------------------------
# 4. MODEL --------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=stoi[PAD])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids, attn_mask):
        x = self.embed(ids)
        key_padding_mask = ~(attn_mask.bool())
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        # mean pooling over non-pad tokens
        lengths = attn_mask.sum(1).unsqueeze(-1)
        pooled = (x * attn_mask.unsqueeze(-1)).sum(1) / lengths
        return self.fc(pooled)


num_classes = len(set(dsets["train"]["label"]))
model = TinyTransformer(vocab_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------
# 5. TRAIN / EVAL LOOP --------------------------------------------------------
def run_epoch(loader, train=False):
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
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


epochs = 6
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss, train_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)

    experiment_data["SPR"]["epochs"].append(epoch)
    experiment_data["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR"]["metrics"]["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={train_f1:.4f} val_F1={val_f1:.4f}  ({time.time()-t0:.1f}s)"
    )

# ------------------------------------------------------------------
# 6. FINAL TEST EVALUATION -----------------------------------------
test_loss, test_f1, test_preds, test_labels = run_epoch(test_loader, train=False)
print(f"\nTest   : loss={test_loss:.4f} Macro_F1={test_f1:.4f}")

experiment_data["SPR"]["predictions"] = test_preds
experiment_data["SPR"]["ground_truth"] = test_labels
experiment_data["SPR"]["test_loss"] = test_loss
experiment_data["SPR"]["test_macro_f1"] = test_f1

# ------------------------------------------------------------------
# 7. SAVE METRICS & PLOTS ------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

plt.figure(figsize=(6, 4))
plt.plot(
    experiment_data["SPR"]["epochs"],
    experiment_data["SPR"]["metrics"]["train_f1"],
    label="train",
)
plt.plot(
    experiment_data["SPR"]["epochs"],
    experiment_data["SPR"]["metrics"]["val_f1"],
    label="val",
)
plt.xlabel("epoch")
plt.ylabel("Macro F1")
plt.title("SPR Macro-F1")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, "SPR_f1_curve.png")
plt.savefig(plot_path)
print(f"Learning-curve figure saved to {plot_path}")
