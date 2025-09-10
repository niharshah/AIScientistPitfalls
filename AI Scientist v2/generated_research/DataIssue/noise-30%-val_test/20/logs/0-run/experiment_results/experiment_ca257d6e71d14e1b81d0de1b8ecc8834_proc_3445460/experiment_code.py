import os, pathlib, math, time, random, string, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---- working dir / device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---- load SPR_BENCH (fall back to toy synth) --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _load(f"{split}.csv")
    return dd


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:  # tiny synthetic demo if real data unavailable
    print("SPR_BENCH not found – building synthetic toy data")

    def synth(n, start_id=0):
        rows = []
        for i in range(n):
            L = random.randint(5, 17)
            seq = "".join(random.choices(string.ascii_uppercase[:12], k=L))
            label = int(seq.count("A") % 2 == 0)
            rows.append(
                {"id": start_id + i, "sequence": seq, "label": label, "complexity": L}
            )
        return rows

    spr = DatasetDict()
    for split, n in zip(["train", "dev", "test"], [4000, 800, 800]):
        spr[split] = load_dataset("json", data_files={"train": synth(n)}, split="train")

print({k: len(v) for k, v in spr.items()})

# ---- vocabulary & encoding ---------------------------------------------------
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
print("Vocab size", vocab_size)


def encode(seq, max_len):
    ids = [vocab["<cls>"]] + [vocab.get(c, 1) for c in seq][: max_len - 1]
    ids = ids + ([0] * (max_len - len(ids)))
    return ids


MAX_LEN = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)


# ---- dataset to torch --------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        seq_ids = torch.tensor(encode(ex["sequence"], MAX_LEN), dtype=torch.long)
        label = torch.tensor(int(ex["label"]), dtype=torch.long)
        if "complexity" in ex:
            w = float(ex["complexity"])
        else:
            w = float(len(ex["sequence"]))
        return {"input_ids": seq_ids, "labels": label, "weights": torch.tensor(w)}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = map(SPRDataset, (spr["train"], spr["dev"], spr["test"]))


# ---- sinusoidal positional enc ----------------------------------------------
def sinusoid_position_encoding(L, D, device):
    pe = torch.zeros(L, D, device=device)
    pos = torch.arange(0, L, device=device).float()[:, None]
    i = torch.arange(0, D, device=device).float()[None, :]
    angle = pos / torch.pow(10000, (2 * (i // 2)) / D)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


# ---- model -------------------------------------------------------------------
class CharTransformer(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=128, nhead=8, layers=4, ff_dim=256, classes=2
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.register_buffer(
            "pos_pe", sinusoid_position_encoding(MAX_LEN, emb_dim, "cpu")
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, classes)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos_pe[: x.size(1)]
        h = self.encoder(h, src_key_padding_mask=mask)
        cls = self.norm(h[:, 0])
        return self.out(cls)


# ---- complexity weighted accuracy -------------------------------------------
def cwa(preds, labels, weights):
    correct = (preds == labels).astype(float)
    return (correct * weights).sum() / weights.sum()


# ---- dataloaders -------------------------------------------------------------
batch_size = 32
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)

# ---- training setup ----------------------------------------------------------
model = CharTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(reduction="none")  # we'll weight manually
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
epochs = 15
best_cwa = -1
best_state = None
patience = 4
wait = 0

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weights": [],
    }
}

# ---- training loop -----------------------------------------------------------
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    n_items = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        per_ex_loss = criterion(logits, batch["labels"])
        weights = batch["weights"]
        loss = (per_ex_loss * weights).mean()  # weight the loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        n_items += batch["labels"].size(0)
    train_loss = epoch_loss / n_items
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----------------------------------------------------------
    model.eval()
    v_loss = 0
    v_items = 0
    preds_all = []
    labels_all = []
    w_all = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            per_ex = criterion(logits, batch["labels"])
            loss = (per_ex * batch["weights"]).mean()
            v_loss += loss.item() * batch["labels"].size(0)
            v_items += batch["labels"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            w = batch["weights"].cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels)
            w_all.extend(w)
    v_loss /= v_items
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    val_cwa = cwa(np.array(preds_all), np.array(labels_all), np.array(w_all))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": macro_f1, "cwa": val_cwa}
    )
    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | Macro-F1={macro_f1:.4f} | CWA={val_cwa:.4f}"
    )

    # early stopping on CWA
    if val_cwa > best_cwa + 1e-4:
        best_cwa = val_cwa
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping triggered.")
        break

# ---- restore best model ------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

# ---- final test evaluation ---------------------------------------------------
model.eval()
preds_test = []
labels_test = []
w_test = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"])
        preds = logits.argmax(1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        w = batch["weights"].cpu().numpy()
        preds_test.extend(preds)
        labels_test.extend(labels)
        w_test.extend(w)
macro_f1_test = f1_score(labels_test, preds_test, average="macro")
cwa_test = cwa(np.array(preds_test), np.array(labels_test), np.array(w_test))
print(f"TEST: Macro-F1={macro_f1_test:.4f} | CWA={cwa_test:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds_test
experiment_data["SPR_BENCH"]["ground_truth"] = labels_test
experiment_data["SPR_BENCH"]["weights"] = w_test

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
