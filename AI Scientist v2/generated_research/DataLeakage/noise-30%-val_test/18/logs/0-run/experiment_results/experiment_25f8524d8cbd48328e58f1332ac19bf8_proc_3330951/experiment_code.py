import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ------------------ reproducibility -----------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ------------------ device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ load SPR_BENCH ------------------
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
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:
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
    dsets["train"] = synth_split(512)
    dsets["dev"] = synth_split(128)
    dsets["test"] = synth_split(128)
print({k: len(v) for k, v in dsets.items()})

# ------------------ vocab & encoding ----------------
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


# ------------------ collate fn ----------------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


# ------------------ model ---------------------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        out, _ = self.lstm(em)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)


# ------------------ training routine ---------------
def train_with_bs(batch_size, epochs=5, lr=1e-3):
    # loaders
    train_loader = DataLoader(
        dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    # model / optim
    model = SPRClassifier(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "test_f1": None,
    }
    best_f1 = 0.0
    best_preds, best_gts = None, None

    for epoch in range(1, epochs + 1):
        # train
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
        tr_f1 = f1_score(tr_gts, tr_preds, average="macro")

        # dev
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
        dv_f1 = f1_score(dv_gts, dv_preds, average="macro")

        print(
            f"[bs={batch_size}] Epoch {epoch}: train_loss={np.mean(tr_losses):.4f}, "
            f"val_loss={np.mean(dv_losses):.4f}, val_macroF1={dv_f1:.4f}"
        )

        # log
        log["metrics"]["train"].append(tr_f1)
        log["metrics"]["val"].append(dv_f1)
        log["losses"]["train"].append(np.mean(tr_losses))
        log["losses"]["val"].append(np.mean(dv_losses))
        log["epochs"].append(epoch)

        if dv_f1 > best_f1:
            best_f1 = dv_f1
            best_preds, best_gts = dv_preds, dv_gts

    # test eval
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
    log["predictions"] = best_preds
    log["ground_truth"] = best_gts
    log["test_f1"] = test_f1
    print(
        f"[bs={batch_size}] Best Dev Macro-F1={best_f1:.4f} | Test Macro-F1={test_f1:.4f}"
    )
    return log


# ------------------ hyperparameter sweep -----------
batch_sizes = [32, 64, 128]
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

for bs in batch_sizes:
    log = train_with_bs(bs)
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = log

# ------------------ save ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
