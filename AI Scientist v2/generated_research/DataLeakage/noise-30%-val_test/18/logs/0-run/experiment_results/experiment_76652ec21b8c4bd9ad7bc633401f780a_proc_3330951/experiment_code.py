import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------------- paths & dirs --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- dataset -------------------------
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
else:  # ------------ synthetic fallback -------------
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

# ---------------- vocab & encoding ----------------
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


# ---------------- collate fn ----------------------
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


# ---------------- model def -----------------------
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


# ---------------- experiment store ----------------
experiment_data = {"embed_dim": {}}

# ---------------- hyper-parameter grid ------------
embed_dims = [32, 64, 128, 256]
overall_best_f1, overall_best_cfg, overall_best_preds, overall_best_gts = (
    -1,
    None,
    None,
    None,
)

# ---------------- training loop per embed_dim -----
epochs = 5
for dim in embed_dims:
    print(f"\n===== Training with embed_dim={dim} =====")
    model = SPRClassifier(vocab_size, embed_dim=dim).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    edata = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    best_dev_f1, best_dev_preds, best_dev_gts = -1, None, None

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_losses, tr_preds, tr_gts = [], [], []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            opt.step()
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
                loss = crit(logits, batch["labels"])
                dv_losses.append(loss.item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dev_f1 = f1_score(dv_gts, dv_preds, average="macro")

        print(
            f"Epoch {epoch}: train_loss={np.mean(tr_losses):.4f}  "
            f"val_loss={np.mean(dv_losses):.4f}  val_macroF1={dev_f1:.4f}"
        )

        # store epoch data
        edata["metrics"]["train"].append(train_f1)
        edata["metrics"]["val"].append(dev_f1)
        edata["losses"]["train"].append(np.mean(tr_losses))
        edata["losses"]["val"].append(np.mean(dv_losses))
        edata["epochs"].append(epoch)

        # keep best epoch per dim
        if dev_f1 > best_dev_f1:
            best_dev_f1, best_dev_preds, best_dev_gts = (
                dev_f1,
                dv_preds.copy(),
                dv_gts.copy(),
            )

    # save best preds/gts for this dim
    edata["predictions"] = best_dev_preds
    edata["ground_truth"] = best_dev_gts
    experiment_data["embed_dim"][dim] = edata

    # track best overall cfg
    if best_dev_f1 > overall_best_f1:
        overall_best_f1 = best_dev_f1
        overall_best_cfg = (dim, model)  # keep model ref for test
        overall_best_preds, overall_best_gts = best_dev_preds, best_dev_gts

# ---------------- final test on best cfg ----------
best_dim, best_model = overall_best_cfg
print(f"\n*** Best embed_dim={best_dim} with Dev Macro-F1={overall_best_f1:.4f} ***")
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
print(f"Test Macro-F1 with best embed_dim={best_dim}: {test_f1:.4f}")

# ---------------- save experiment data ------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
