import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
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
weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
experiment_data = {}
epochs = 5
criterion = nn.CrossEntropyLoss()

for wd in weight_decay_values:
    key = f"weight_decay_{wd}"
    experiment_data[key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = CharGRU(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    print(f"\n--- Training with weight_decay={wd} ---")
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        total_loss, items = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            items += batch["labels"].size(0)
        train_loss = total_loss / items
        experiment_data[key]["SPR_BENCH"]["losses"]["train"].append(train_loss)
        # validation
        model.eval()
        val_loss, val_items = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item() * batch["labels"].size(0)
                val_items += batch["labels"].size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        val_loss /= val_items
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        experiment_data[key]["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data[key]["SPR_BENCH"]["metrics"]["val"].append(macro_f1)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, Macro-F1={macro_f1:.4f}")
    # store final epoch preds/labels
    experiment_data[key]["SPR_BENCH"]["predictions"] = all_preds
    experiment_data[key]["SPR_BENCH"]["ground_truth"] = all_labels
    # free memory
    del model, optimizer
    torch.cuda.empty_cache()

# -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
