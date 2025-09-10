import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt
from typing import List, Dict

# ---------- paths / saving ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load (real or synthetic) SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def get_dataset() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print(f"Loading real SPR_BENCH from {p}")
            return load_spr_bench(p)
    print("SPR_BENCH not found, creating synthetic toy dataset")

    def synth(n):
        rows = []
        shapes = "ABCD"
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 12)))
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def to_ds(rows):
        return load_dataset(
            "json", data_files={"data": rows}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"] = to_ds(synth(2000))
    d["dev"] = to_ds(synth(500))
    d["test"] = to_ds(synth(500))
    return d


spr = get_dataset()

# ---------- vocab / encoding ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(list(set(all_text)))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi) + 1
print("Vocab size", vocab_size - 1)
max_len = min(100, max(len(s) for s in spr["train"]["sequence"]))


def encode(seq: str) -> List[int]:
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.y = hf_split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.float),
        }


# ---------- model ----------
class CharBiGRU(nn.Module):
    def __init__(self, vocab_sz: int, emb_dim: int = 64, hid: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h).squeeze(1)


# ---------- experiment data ----------
experiment_data = {"batch_size": {}}


def run_experiment(bs: int, epochs: int = 5, lr: float = 1e-3):
    print(f"\n=== Training with batch_size={bs} ===")
    train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=bs, shuffle=True)
    val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=bs)
    test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=bs)

    model = CharBiGRU(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    record = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_losses = []
        tr_preds = []
        tr_lbls = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            tr_lbls.extend(batch["label"].long().cpu().numpy())
        train_f1 = f1_score(tr_lbls, tr_preds, average="macro")

        # validation
        model.eval()
        val_losses = []
        val_preds = []
        val_lbls = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                val_losses.append(criterion(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
                val_lbls.extend(batch["label"].long().cpu().numpy())
        val_f1 = f1_score(val_lbls, val_preds, average="macro")
        print(f"Epoch {epoch}: val_loss {np.mean(val_losses):.4f}, val_f1 {val_f1:.4f}")

        record["metrics"]["train_macro_f1"].append(train_f1)
        record["metrics"]["val_macro_f1"].append(val_f1)
        record["losses"]["train"].append(np.mean(tr_losses))
        record["losses"]["val"].append(np.mean(val_losses))
        record["epochs"].append(epoch)

    # test
    model.eval()
    test_preds = []
    test_lbls = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            test_preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
            test_lbls.extend(batch["label"].long().cpu().numpy())
    record["predictions"] = test_preds
    record["ground_truth"] = test_lbls
    test_f1 = f1_score(test_lbls, test_preds, average="macro")
    print(f"Test Macro-F1 for bs={bs}: {test_f1:.4f}")

    # plot
    plt.figure()
    plt.plot(record["epochs"], record["losses"]["train"], label="train_loss")
    plt.plot(record["epochs"], record["losses"]["val"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss curve bs={bs}")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_loss_curve_bs{bs}.png"))
    plt.close()

    return record


for bs in [32, 64, 256]:
    experiment_data["batch_size"][f"SPR_BENCH_bs{bs}"] = run_experiment(bs)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
