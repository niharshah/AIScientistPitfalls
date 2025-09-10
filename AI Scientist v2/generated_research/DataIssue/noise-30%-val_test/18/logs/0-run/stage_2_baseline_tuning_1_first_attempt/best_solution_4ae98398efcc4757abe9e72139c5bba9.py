import os, pathlib, random, string, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ----------------- misc setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ----------------- load SPR_BENCH ------------------
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

    dsets = DatasetDict(
        {"train": synth_split(512), "dev": synth_split(128), "test": synth_split(128)}
    )
print({k: len(v) for k, v in dsets.items()})

# ----------------- vocab & encoding ----------------
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


# ----------------- dataloader helpers --------------
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


# ----------------- model ---------------------------
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


# ----------------- hyper-parameter sweep -----------
weight_decays = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
epochs = 5

experiment_data = {"weight_decay": {"SPR_BENCH": {}}}

for wd in weight_decays:
    key = str(wd)
    experiment_data["weight_decay"]["SPR_BENCH"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "test_f1": None,
    }

    model = SPRClassifier(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    best_f1, best_preds, best_gts = 0.0, [], []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_losses, tr_preds, tr_gts = [], [], []
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
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
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                dv_losses.append(loss.item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dev_f1 = f1_score(dv_gts, dv_preds, average="macro")
        print(
            f"[wd={wd:.0e}] Epoch {epoch}: train_loss={np.mean(tr_losses):.4f}, "
            f"val_loss={np.mean(dv_losses):.4f}, val_F1={dev_f1:.4f}"
        )

        # ---- log ----
        dstore = experiment_data["weight_decay"]["SPR_BENCH"][key]
        dstore["metrics"]["train"].append(train_f1)
        dstore["metrics"]["val"].append(dev_f1)
        dstore["losses"]["train"].append(np.mean(tr_losses))
        dstore["losses"]["val"].append(np.mean(dv_losses))
        dstore["epochs"].append(epoch)

        # best preds
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_preds, best_gts = dv_preds, dv_gts

    # ---- final test evaluation ----
    model.eval()
    tst_preds, tst_gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            tst_preds.extend(logits.argmax(1).cpu().numpy())
            tst_gts.extend(batch["labels"].cpu().numpy())
    test_f1 = f1_score(tst_gts, tst_preds, average="macro")
    print(f"[wd={wd:.0e}] Best Dev F1={best_f1:.4f} | Test F1={test_f1:.4f}")

    # ---- save best preds ----
    dstore["predictions"] = best_preds
    dstore["ground_truth"] = best_gts
    dstore["test_f1"] = test_f1

# ----------------- save ----------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All experiment data saved.")
