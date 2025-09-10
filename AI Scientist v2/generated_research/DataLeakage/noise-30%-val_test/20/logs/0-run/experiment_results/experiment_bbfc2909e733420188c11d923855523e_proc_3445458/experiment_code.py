import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
# workspace / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# load SPR_BENCH or synthesize dummy data
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_data = DATA_PATH.exists()

if have_data:
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found – generating small synthetic dataset.")

    def synth_split(n, start_id=0):
        rows = []
        for i in range(n):
            seq_len = random.randint(5, 15)
            seq = "".join(random.choices(string.ascii_uppercase[:12], k=seq_len))
            label = int(seq.count("A") % 2 == 0)
            rows.append(
                {
                    "id": start_id + i,
                    "sequence": seq,
                    "label": label,
                    "complexity": seq_len % 4 + 1,
                }
            )
        return rows

    spr = DatasetDict()
    spr["train"] = load_dataset(
        "json", data_files={"train": synth_split(4000)}, split="train"
    )
    spr["dev"] = load_dataset(
        "json", data_files={"train": synth_split(800, 4000)}, split="train"
    )
    spr["test"] = load_dataset(
        "json", data_files={"train": synth_split(800, 4800)}, split="train"
    )

print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------
# build vocabulary incl. <pad>, <unk>, <cls>
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq, max_len):
    ids = [vocab["<cls>"]] + [vocab.get(ch, vocab["<unk>"]) for ch in seq][
        : max_len - 1
    ]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


MAX_LEN = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)  # +1 for <cls>


# -------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        seq_ids = torch.tensor(encode(ex["sequence"], MAX_LEN), dtype=torch.long)
        label = torch.tensor(int(ex["label"]), dtype=torch.long)
        weight = torch.tensor(float(ex.get("complexity", 1.0)), dtype=torch.float)
        return {"input_ids": seq_ids, "labels": label, "weights": weight}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = map(
    SPRTorchDataset, (spr["train"], spr["dev"], spr["test"])
)


# -------------------------------------------------
class CharTransformer(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=64, nhead=4, num_layers=2, dim_ff=128, num_classes=2
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_LEN, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        mask = x == vocab["<pad>"]
        h = self.emb(x) + self.pos[: x.size(1)]
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_h = h[:, 0]  # take <cls>
        return self.fc(cls_h)


# -------------------------------------------------
def complexity_weighted_accuracy(preds, labels, weights):
    correct = (preds == labels).astype(float)
    return (correct * weights).sum() / weights.sum()


# -------------------------------------------------
batch_size = 32
epochs = 8
model = CharTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weights": [],
    }
}

for epoch in range(1, epochs + 1):
    # ---- training ----
    model.train()
    tot_loss, tot_items = 0.0, 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        tot_items += batch["labels"].size(0)
    train_loss = tot_loss / tot_items
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # ---- validation ----
    model.eval()
    val_loss, val_items = 0.0, 0
    all_preds, all_labels, all_w = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_items += batch["labels"].size(0)
            preds = logits.argmax(1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            weights = batch["weights"].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_w.extend(weights.tolist())
    val_loss /= val_items
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cwa = complexity_weighted_accuracy(
        np.array(all_preds), np.array(all_labels), np.array(all_w)
    )
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": macro_f1, "cwa": cwa}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | Macro-F1={macro_f1:.4f} | CWA={cwa:.4f}"
    )
    scheduler.step()

# store last predictions
experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_labels
experiment_data["SPR_BENCH"]["weights"] = all_w

# -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
