import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------------------- dirs / device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------- experiment dict --------------------------
experiment_data = {
    "count_only": {
        "spr_bench": {
            "epochs": [],
            "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
            "losses": {"train": [], "val": [], "test": None},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

ed = experiment_data["count_only"]["spr_bench"]


# ---------------------------- dataset ----------------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})
num_labels = len(set(spr["train"]["label"]))

# ---------------------------- vocab & encode ---------------------------
special_tokens = ["<PAD>", "<UNK>", "<SOS>"]
char_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)
pad_id, unk_id, sos_id = char_vocab["<PAD>"], char_vocab["<UNK>"], char_vocab["<SOS>"]


def encode(example):
    seq = example["sequence"]
    counts = np.zeros(len(char_vocab), dtype=np.int16)
    for ch in seq:
        counts[char_vocab.get(ch, unk_id)] += 1
    return {"count_vec": counts.tolist()}


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(encode, remove_columns=[])


# ---------------------------- collate ----------------------------------
def collate(batch):
    count_tensor = torch.stack(
        [torch.tensor(b["count_vec"], dtype=torch.float32) for b in batch]
    )
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"count_vec": count_tensor, "labels": labels}


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    for split in ["train", "dev", "test"]
}


# ---------------------------- model ------------------------------------
class CountOnlyClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, d_model=256, dropout=0.2):
        super().__init__()
        self.count_proj = nn.Sequential(
            nn.Linear(vocab_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, count_vec):
        rep = self.count_proj(count_vec)
        return self.classifier(rep)


model = CountOnlyClassifier(len(char_vocab), num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


# ---------------------------- train / eval -----------------------------
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        count_vec = batch["count_vec"].to(device)
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(count_vec)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tot_loss += loss.item() * labels.size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(labels.cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------------------------- training loop ----------------------------
best_val_f1, patience, wait = 0.0, 3, 0
max_epochs = 15
save_path = os.path.join(working_dir, "count_only_best.pt")

for epoch in range(1, max_epochs + 1):
    tic = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(loaders["train"], train=True)
    val_loss, val_f1, _, _ = run_epoch(loaders["dev"], train=False)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_F1={val_f1:.4f}")
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
    print(f"  time {time.time()-tic:.1f}s  best_val_F1={best_val_f1:.4f}")

# ---------------------------- testing ----------------------------------
model.load_state_dict(torch.load(save_path))
test_loss, test_f1, test_preds, test_gts = run_epoch(loaders["test"], train=False)
print(f"Test Macro F1: {test_f1:.4f}")
ed["losses"]["test"] = test_loss
ed["metrics"]["test_f1"] = test_f1
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# ---------------------------- save data --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
