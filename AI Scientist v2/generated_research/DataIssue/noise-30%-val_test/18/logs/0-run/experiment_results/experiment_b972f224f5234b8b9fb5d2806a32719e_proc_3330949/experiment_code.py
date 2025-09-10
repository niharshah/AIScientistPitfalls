import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------- utils / dirs ------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR_BENCH ----------
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
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:
    # ---------- synthetic fallback ----
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = "".join(
                random.choices(list(string.ascii_lowercase) + ["#", "@", "&"], k=L)
            )
            lbl = int(seq.count("#") % 2 == 0)
            seqs.append(seq)
            labels.append(lbl)
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

# ---------- vocab & encoding ---------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):  # char -> ids
    return [vocab.get(ch, vocab[UNK]) for ch in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------- dataloaders --------------
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


# ---------- model --------------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        out, _ = self.lstm(em)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)


# ---------- experiment_data ----------
experiment_data = {"num_lstm_layers": {"SPR_BENCH": {}}}


# ---------- training / evaluation ----
def train_eval(num_layers, epochs=5, lr=1e-3):
    model = SPRClassifier(vocab_size, num_layers=num_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logs = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    best_f1, best_preds, best_gts = 0.0, None, None

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_losses, tr_preds, tr_gts = [], [], []
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_gts.extend(batch["labels"].cpu().numpy())
        tr_f1 = f1_score(tr_gts, tr_preds, average="macro")

        # ---- dev ----
        model.eval()
        dv_losses, dv_preds, dv_gts = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                dv_losses.append(loss.item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dv_f1 = f1_score(dv_gts, dv_preds, average="macro")

        print(
            f"[layers={num_layers}] Epoch {ep}: train_loss={np.mean(tr_losses):.4f}, "
            f"val_loss={np.mean(dv_losses):.4f}, val_macroF1={dv_f1:.4f}"
        )

        # logging
        logs["metrics"]["train"].append(tr_f1)
        logs["metrics"]["val"].append(dv_f1)
        logs["losses"]["train"].append(np.mean(tr_losses))
        logs["losses"]["val"].append(np.mean(dv_losses))
        logs["epochs"].append(ep)
        if dv_f1 > best_f1:
            best_f1, best_preds, best_gts = dv_f1, dv_preds.copy(), dv_gts.copy()

    # ---- test with best model (simple retraining not saved; evaluate last weights) ----
    model.eval()
    tst_preds, tst_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            tst_preds.extend(logits.argmax(1).cpu().numpy())
            tst_gts.extend(batch["labels"].cpu().numpy())
    tst_f1 = f1_score(tst_gts, tst_preds, average="macro")
    print(f"[layers={num_layers}] Best Dev F1={best_f1:.4f} | Test F1={tst_f1:.4f}")

    logs["predictions"], logs["ground_truth"] = best_preds, best_gts
    logs["best_dev_f1"], logs["test_f1"] = best_f1, tst_f1
    return logs, best_f1


# ---------- hyperparameter tuning ----
layer_options = [1, 2, 3]
best_overall = -1.0
best_setting = None
for n_layers in layer_options:
    result, dev_f1 = train_eval(n_layers)
    experiment_data["num_lstm_layers"]["SPR_BENCH"][str(n_layers)] = result
    if dev_f1 > best_overall:
        best_overall, best_setting = dev_f1, n_layers

print(f"\nBest setting: num_layers={best_setting} with Dev F1={best_overall:.4f}")

# ---------- save experiment data ----
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
