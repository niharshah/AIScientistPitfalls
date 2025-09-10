import os, pathlib, random, string, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# folders / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------------------
# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ------------------------------------------------------------------
# load or synthesize SPR_BENCH
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for sp in ("train", "dev", "test"):
        dd[sp] = _load(f"{sp}.csv")
    return dd


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_data = DATA_PATH.exists()
if have_data:
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH not found, generating synthetic dataset.")

    def synth_split(n):
        rows = []
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:10], k=random.randint(5, 15))
            )
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def hf_from(rows):
        return load_dataset("json", data_files={"data": [rows]}, split="train")

    spr = DatasetDict()
    spr["train"] = hf_from(synth_split(2000))
    spr["dev"] = hf_from(synth_split(400))
    spr["test"] = hf_from(synth_split(400))
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# vocab + encoding helpers
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq, max_len):
    ids = [vocab.get(ch, 1) for ch in seq][:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


max_len = min(max(len(ex["sequence"]) for ex in spr["train"]), 120)


# ------------------------------------------------------------------
# pytorch datasets
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(
                encode(ex["sequence"], max_len), dtype=torch.long
            ),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
        }


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# model that supports variable layer depth
class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, num_classes=2, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        logits = self.fc(h[-1])
        return logits


# ------------------------------------------------------------------
# training routine
def train_model(num_layers, epochs=5, lr=1e-3):
    model = CharGRU(vocab_size, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    final_preds, final_labels = None, None
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot_loss, items = 0, 0
        tr_preds, tr_labels = [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            items += batch["labels"].size(0)
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_labels.extend(batch["labels"].cpu().numpy())
        train_loss = tot_loss / items
        train_f1 = f1_score(tr_labels, tr_preds, average="macro")
        losses["train"].append(train_loss)
        metrics["train"].append(train_f1)
        # ---- validation ----
        model.eval()
        tot_loss, items = 0, 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                tot_loss += loss.item() * batch["labels"].size(0)
                items += batch["labels"].size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_labels.extend(batch["labels"].cpu().numpy())
        val_loss = tot_loss / items
        val_f1 = f1_score(v_labels, v_preds, average="macro")
        losses["val"].append(val_loss)
        metrics["val"].append(val_f1)
        print(
            f"[layers={num_layers}] Ep {ep}: train F1 {train_f1:.3f} | val F1 {val_f1:.3f}"
        )
        final_preds, final_labels = v_preds, v_labels
    return {
        "metrics": metrics,
        "losses": losses,
        "predictions": final_preds,
        "ground_truth": final_labels,
    }


# ------------------------------------------------------------------
# hyper-parameter sweep: 1,2,3 GRU layers
experiment_data = {"num_gru_layers": {"SPR_BENCH": {}}}
for nl in (1, 2, 3):
    set_seed(42 + nl)  # different but reproducible
    result = train_model(num_layers=nl, epochs=5)
    experiment_data["num_gru_layers"]["SPR_BENCH"][f"layers_{nl}"] = result

# ------------------------------------------------------------------
# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All results saved to", os.path.join(working_dir, "experiment_data.npy"))
