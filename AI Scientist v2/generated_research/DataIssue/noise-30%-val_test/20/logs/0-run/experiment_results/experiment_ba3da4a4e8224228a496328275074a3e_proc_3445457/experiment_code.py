import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from datasets import load_dataset, DatasetDict

# --------------------------------------------------------------------
# Mandatory working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------
# ------------- helper: load SPR_BENCH or build synthetic -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    # ---------- quick synthetic generator with a 'complexity' column ----------
    def synth_split(n, start_id=0):
        rows = []
        for i in range(start_id, start_id + n):
            seq_len = random.randint(5, 15)
            seq = "".join(random.choices(string.ascii_uppercase[:12], k=seq_len))
            complexity = random.randint(1, 5)  # pretend #atomic predicates
            # a 2-predicate xor rule as a toy
            even_a = seq.count("A") % 2 == 0
            has_c = "C" in seq
            label = int(even_a ^ has_c)
            rows.append(
                {"id": i, "sequence": seq, "label": label, "complexity": complexity}
            )
        return rows

    print("SPR_BENCH not found – creating synthetic data.")
    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": synth_split(4000)}, split="train"
            ),
            "dev": load_dataset(
                "json", data_files={"train": synth_split(800, 4000)}, split="train"
            ),
            "test": load_dataset(
                "json", data_files={"train": synth_split(800, 4800)}, split="train"
            ),
        }
    )
print({k: len(v) for k, v in spr.items()})

# --------------------------------------------------------------------
# ------------------------- vocabulary --------------------------------
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


# --------------------------------------------------------------------
# ---------------------- Torch Dataset wrapper ------------------------
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.has_complex = "complexity" in hf_dataset.column_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        w = ex["complexity"] if "complexity" in ex else 1.0
        return {
            "input_ids": torch.tensor(
                encode(ex["sequence"], max_len), dtype=torch.long
            ),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
            "weight": torch.tensor(float(w), dtype=torch.float),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "weights": torch.stack([b["weight"] for b in batch]),
    }


train_ds, dev_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)
batch_size = 32
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# --------------------------------------------------------------------
# ---------------------------- Model ----------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab, emb_dim=128, nhead=4, nlayers=2, nclass=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim * 2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(emb_dim, nclass)

    def forward(self, x):
        mask = x == 0  # pad mask
        emb = self.emb(x).transpose(0, 1)  # (S,B,E) for transformer
        enc = self.encoder(emb, src_key_padding_mask=mask).transpose(
            0, 1
        )  # back to (B,S,E)
        pooled = enc.mean(dim=1)
        return self.fc(pooled)


model = TinyTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------------------------------------------------------------
# ------------------ Metric: Complexity-Weighted Accuracy -------------
def complexity_weighted_accuracy(preds, labels, weights):
    correct = (preds == labels).astype(float)
    return (correct * weights).sum() / weights.sum()


# --------------------------------------------------------------------
# --------------- experiment data container ---------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"accuracy": [], "cwa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weights": [],
    }
}

# --------------------------------------------------------------------
# -------------------------- Training ---------------------------------
epochs = 10
best_cwa, patience, wait = 0.0, 3, 0

for epoch in range(1, epochs + 1):
    # ----------------- train -----------------
    model.train()
    running_loss, n_items = 0.0, 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
        n_items += batch["labels"].size(0)
    train_loss = running_loss / n_items
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ----------------- validation -----------------
    model.eval()
    val_loss, n_val = 0.0, 0
    all_preds, all_labels, all_w = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch_cpu = {k: v for k, v in batch.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            n_val += batch["labels"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            labels = batch_cpu["labels"].numpy()
            wts = batch_cpu["weights"].numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_w.extend(wts.tolist())
    val_loss /= n_val
    acc = accuracy_score(all_labels, all_preds)
    cwa = complexity_weighted_accuracy(
        np.array(all_preds), np.array(all_labels), np.array(all_w)
    )
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["accuracy"].append(acc)
    experiment_data["SPR_BENCH"]["metrics"]["cwa"].append(cwa)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, Acc={acc:.4f}, CWA={cwa:.4f}")

    # early stopping on best CWA
    if cwa > best_cwa:
        best_cwa, wait = cwa, 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# --------------------------------------------------------------------
# -------- evaluate on test set with best saved parameters -----------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.eval()
test_preds, test_labels, test_w = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch_cpu = {k: v for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        preds = logits.argmax(1).cpu().numpy()
        test_preds.extend(preds.tolist())
        test_labels.extend(batch_cpu["labels"].numpy().tolist())
        test_w.extend(batch_cpu["weights"].numpy().tolist())
test_acc = accuracy_score(test_labels, test_preds)
test_cwa = complexity_weighted_accuracy(
    np.array(test_preds), np.array(test_labels), np.array(test_w)
)
print(f"\nTest Accuracy = {test_acc:.4f}, Test CWA = {test_cwa:.4f}")

# store predictions/gt/weights
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
experiment_data["SPR_BENCH"]["weights"] = test_w

# --------------------------------------------------------------------
# ---------------- save experiment data -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f'Experiment data saved to {os.path.join(working_dir, "experiment_data.npy")}')
