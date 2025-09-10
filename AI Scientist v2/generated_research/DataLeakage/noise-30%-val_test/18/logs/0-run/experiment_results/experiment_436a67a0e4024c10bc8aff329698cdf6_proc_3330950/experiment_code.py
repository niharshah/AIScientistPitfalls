import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------------- filesystem / save dir -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- load SPR_BENCH ------------------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    dd["train"] = _load("train.csv")
    dd["dev"] = _load("dev.csv")
    dd["test"] = _load("test.csv")
    return dd


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:
    # fallback very small synthetic dataset
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = "".join(
                random.choices(list(string.ascii_lowercase) + ["#", "@", "&"], k=L)
            )
            labels.append(int(seq.count("#") % 2 == 0))
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict()
    dsets["train"] = synth_split(512)
    dsets["dev"] = synth_split(128)
    dsets["test"] = synth_split(128)
print({k: len(v) for k, v in dsets.items()})

# ---------------- vocab / encoding ----------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(ch, vocab[UNK]) for ch in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------------- collator / loaders --------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


batch_size = 128
train_loader = DataLoader(
    dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------------- model definition ----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        lstm_out, _ = self.lstm(em)
        pooled, _ = torch.max(lstm_out, dim=1)
        return self.fc(pooled)


# ---------------- hyperparameter sweep ------------
learning_rates = [5e-4, 1e-3, 2e-3, 5e-3]
epochs = 5

experiment_data = {"learning_rate": {}}

for lr in learning_rates:
    print(f"\n=== Training with learning rate {lr} ===")
    model = SPRClassifier(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_metrics = {"train": [], "val": []}
    run_losses = {"train": [], "val": []}
    best_f1, best_preds, best_gts = 0.0, [], []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_losses, tr_preds, tr_gts = [], [], []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_gts.extend(batch["labels"].cpu().numpy())
        train_f1 = f1_score(tr_gts, tr_preds, average="macro")

        # ---- dev ----
        model.eval()
        dv_losses, dv_preds, dv_gts = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                dv_losses.append(criterion(logits, batch["labels"]).item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dev_f1 = f1_score(dv_gts, dv_preds, average="macro")

        run_metrics["train"].append(train_f1)
        run_metrics["val"].append(dev_f1)
        run_losses["train"].append(np.mean(tr_losses))
        run_losses["val"].append(np.mean(dv_losses))

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_preds, best_gts = dv_preds.copy(), dv_gts.copy()

        print(
            f"Epoch {epoch}: lr={lr} train_loss={np.mean(tr_losses):.4f} "
            f"val_loss={np.mean(dv_losses):.4f} val_macroF1={dev_f1:.4f}"
        )

    # ---- final test evaluation ----
    model.eval()
    tst_preds, tst_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            tst_preds.extend(logits.argmax(1).cpu().numpy())
            tst_gts.extend(batch["labels"].cpu().numpy())
    test_f1 = f1_score(tst_gts, tst_preds, average="macro")
    print(f"LR {lr}: Best Dev F1 = {best_f1:.4f} | Test F1 = {test_f1:.4f}")

    # ---- store experiment data ----
    exp_key = f"lr_{lr}"
    experiment_data["learning_rate"][exp_key] = {
        "metrics": run_metrics,
        "losses": run_losses,
        "predictions": best_preds,
        "ground_truth": best_gts,
        "best_dev_f1": best_f1,
        "test_f1": test_f1,
        "epochs": list(range(1, epochs + 1)),
    }

# ---------------- save all experiment data --------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
