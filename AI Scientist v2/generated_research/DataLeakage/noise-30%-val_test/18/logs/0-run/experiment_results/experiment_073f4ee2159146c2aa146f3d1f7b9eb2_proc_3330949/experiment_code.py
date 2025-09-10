import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------- experiment data skeleton ----------
experiment_data = {
    "num_epochs": {  # hyperparam tuning type
        "SPR_BENCH": {"configurations": {}}  # will be filled with epochs_5 / 10 / ...
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load data ----------
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
else:  # fallback synthetic data
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
    dsets["train"], dsets["dev"], dsets["test"] = (
        synth_split(512),
        synth_split(128),
        synth_split(128),
    )
print({k: len(v) for k, v in dsets.items()})

# ---------- vocab ----------
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


# ---------- dataloaders ----------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(map(len, ids))
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


batch_size = 128
train_loader = lambda: DataLoader(
    dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        out, _ = self.lstm(em)
        pooled, _ = torch.max(out, 1)
        return self.fc(pooled)


# ---------- training routine ----------
def run_training(max_epochs: int, patience: int = 3):
    model = SPRClassifier(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_metrics, dv_metrics, tr_losses, dv_losses = [], [], [], []
    best_state, best_dev_f1, epochs_no_improve = None, 0.0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        t_losses, t_preds, t_gts = [], [], []
        for batch in train_loader():
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
            t_preds.extend(logits.argmax(1).cpu().numpy())
            t_gts.extend(batch["labels"].cpu().numpy())
        train_f1 = f1_score(t_gts, t_preds, average="macro")

        model.eval()
        d_losses, d_preds, d_gts = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                d_losses.append(criterion(logits, batch["labels"]).item())
                d_preds.extend(logits.argmax(1).cpu().numpy())
                d_gts.extend(batch["labels"].cpu().numpy())
        dev_f1 = f1_score(d_gts, d_preds, average="macro")

        tr_metrics.append(train_f1)
        dv_metrics.append(dev_f1)
        tr_losses.append(np.mean(t_losses))
        dv_losses.append(np.mean(d_losses))
        print(
            f"[epochs={max_epochs}] Epoch {epoch}: train_loss={tr_losses[-1]:.4f} dev_loss={dv_losses[-1]:.4f} dev_F1={dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1, best_state = dev_f1, model.state_dict()
            best_preds, best_gts = d_preds.copy(), d_gts.copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience and epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    # load best state for final test
    model.load_state_dict(best_state)
    model.eval()
    t_preds, t_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            t_preds.extend(logits.argmax(1).cpu().numpy())
            t_gts.extend(batch["labels"].cpu().numpy())
    test_f1 = f1_score(t_gts, t_preds, average="macro")

    return {
        "metrics": {"train": tr_metrics, "val": dv_metrics},
        "losses": {"train": tr_losses, "val": dv_losses},
        "predictions": best_preds,
        "ground_truth": best_gts,
        "best_dev_f1": best_dev_f1,
        "test_f1": test_f1,
        "epochs": list(range(1, len(tr_metrics) + 1)),
    }


# ---------- hyperparameter sweep ----------
epoch_options = [5, 10, 15, 20]
for n_ep in epoch_options:
    result = run_training(n_ep, patience=3)
    exp_key = f"epochs_{n_ep}"
    experiment_data["num_epochs"]["SPR_BENCH"]["configurations"][exp_key] = result
    print(
        f"Finished setting {exp_key}: best_dev_F1={result['best_dev_f1']:.4f}, test_F1={result['test_f1']:.4f}"
    )

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All results saved to", os.path.join(working_dir, "experiment_data.npy"))
