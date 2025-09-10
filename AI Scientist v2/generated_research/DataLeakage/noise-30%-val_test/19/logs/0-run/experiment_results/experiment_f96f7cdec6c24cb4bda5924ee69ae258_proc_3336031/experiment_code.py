import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

# ------------------- working dir -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- device ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- data loading ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print(f"Loading real SPR_BENCH from {p}")
            return load_spr_bench(p)
    # fallback toy data
    print("SPR_BENCH not found: creating synthetic dataset")

    def synth(n):
        rows, shapes = [], "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": i, "sequence": seq, "label": label})
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        to_ds(synth(2000)),
        to_ds(synth(500)),
        to_ds(synth(500)),
    )
    return d


spr = get_dataset()

# ------------------- vocabulary --------------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 reserved for PAD
itos = {i: c for c, i in stoi.items()}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.data = [encode(s) for s in hf_split["sequence"]]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx], dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.float),
        }


def make_loader(split, batch_size=256, shuffle=False):
    ds = SPRDataset(spr[split])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = lambda bsz=256: make_loader("train", bsz, True)
dev_loader = lambda bsz=256: make_loader("dev", bsz, False)
test_loader = lambda bsz=256: make_loader("test", bsz, False)


# ------------------- model -------------------------
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_sz,
        max_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        mask = x == 0  # PAD mask
        h = self.emb(x) + self.pos[:, : x.size(1), :]
        h = self.encoder(h, src_key_padding_mask=mask)
        h = h.mean(dim=1)  # simple mean pooling
        return self.fc(h).squeeze(1)


# ------------------- experiment store --------------
experiment_data = {}

# ------------------- hyper-param sweep -------------
configs = [{"layers": 2, "heads": 4}, {"layers": 4, "heads": 8}]

epochs = 4
criterion = nn.BCEWithLogitsLoss()

for conf in configs:
    tag = f"L{conf['layers']}_H{conf['heads']}"
    print(f"\n===== Training config {tag} =====")
    model = TransformerClassifier(
        vocab_size, max_len, d_model=64, n_heads=conf["heads"], n_layers=conf["layers"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rec = {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        tr_losses, tr_preds, tr_lbls = [], [], []
        for batch in train_loader():
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_lbls.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbls, tr_preds)

        # ---------- validate ----------
        model.eval()
        val_losses, val_preds, val_lbls = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                val_losses.append(criterion(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_lbls.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_lbls, val_preds)

        print(
            f"Epoch {ep}: validation_loss = {np.mean(val_losses):.4f} | "
            f"train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )

        # record
        rec["metrics"]["train_MCC"].append(train_mcc)
        rec["metrics"]["val_MCC"].append(val_mcc)
        rec["losses"]["train"].append(np.mean(tr_losses))
        rec["losses"]["val"].append(np.mean(val_losses))
        rec["epochs"].append(ep)

    # -------------- test evaluation -----------------
    model.eval()
    tst_preds, tst_lbls = [], []
    with torch.no_grad():
        for batch in test_loader():
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            tst_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tst_lbls.extend(batch["label"].cpu().numpy())
    test_mcc = matthews_corrcoef(tst_lbls, tst_preds)
    print(f"Test MCC ({tag}): {test_mcc:.4f}")
    rec["predictions"], rec["ground_truth"] = tst_preds, tst_lbls
    rec["test_MCC"] = test_mcc
    experiment_data[tag] = rec

    # -------- plot losses ----------
    plt.figure(figsize=(6, 4))
    plt.plot(rec["epochs"], rec["losses"]["train"], label="train")
    plt.plot(rec["epochs"], rec["losses"]["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"Loss curve {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_{tag}.png"))
    plt.close()

# ------------------- save all ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
