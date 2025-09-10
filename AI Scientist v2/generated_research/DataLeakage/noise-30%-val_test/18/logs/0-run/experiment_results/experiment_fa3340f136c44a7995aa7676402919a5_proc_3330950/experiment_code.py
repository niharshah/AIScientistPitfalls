import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ------------ experiment_data dict -------------
experiment_data = {
    "dropout_prob": {"SPR_BENCH": {}}  # will be filled with one sub-dict per p
}

# ------------ device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------ small helpers --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ------------ load SPR_BENCH or synth ----------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for i in range(n):
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
    dsets["train"], dsets["dev"], dsets["test"] = (
        synth_split(512),
        synth_split(128),
        synth_split(128),
    )
print({k: len(v) for k, v in dsets.items()})

# ------------ vocab / encoding -----------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        vocab.setdefault(ch, len(vocab))


def encode(seq):
    return [vocab.get(ch, vocab[UNK]) for ch in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ------------- collator ------------------------
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


# ------------- model definition ----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, dropout_p=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        out, _ = self.lstm(em)
        pooled, _ = torch.max(out, dim=1)
        pooled = self.drop(pooled)
        return self.fc(pooled)


# ------------- training & tuning ---------------
dropout_vals = [0.0, 0.2, 0.4, 0.6]
epochs = 5
best_overall_f1, best_overall_p, best_state = 0.0, None, None

for p in dropout_vals:
    print(f"\n=== training with dropout_p={p} ===")
    exp_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    experiment_data["dropout_prob"]["SPR_BENCH"][str(p)] = exp_dict

    model = SPRClassifier(len(vocab), dropout_p=p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = 0.0
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
        # ---- eval ----
        model.eval()
        dv_losses, dv_preds, dv_gts = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                dv_losses.append(loss.item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dev_f1 = f1_score(dv_gts, dv_preds, average="macro")
        print(
            f"  Epoch {epoch}: train_loss={np.mean(tr_losses):.4f}, "
            f"val_loss={np.mean(dv_losses):.4f}, val_macroF1={dev_f1:.4f}"
        )
        # record
        exp_dict["metrics"]["train"].append(train_f1)
        exp_dict["metrics"]["val"].append(dev_f1)
        exp_dict["losses"]["train"].append(np.mean(tr_losses))
        exp_dict["losses"]["val"].append(np.mean(dv_losses))
        exp_dict["epochs"].append(epoch)
        # keep best within this p
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            exp_dict["predictions"] = dv_preds
            exp_dict["ground_truth"] = dv_gts
            # also check global best
            if dev_f1 > best_overall_f1:
                best_overall_f1, best_overall_p = dev_f1, p
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # cleanup
    del model
    torch.cuda.empty_cache()

print(
    f"\nBest dev macro-F1 = {best_overall_f1:.4f} achieved with dropout_p={best_overall_p}"
)

# ------------- test evaluation of best ----------
best_model = SPRClassifier(len(vocab), dropout_p=best_overall_p).to(device)
best_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
best_model.eval()
test_preds, test_gts = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"])
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_gts.extend(batch["labels"].cpu().numpy())
test_f1 = f1_score(test_gts, test_preds, average="macro")
print(f"Test macro-F1 with best dropout_p={best_overall_p}: {test_f1:.4f}")

# ------------- save experiment data -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
