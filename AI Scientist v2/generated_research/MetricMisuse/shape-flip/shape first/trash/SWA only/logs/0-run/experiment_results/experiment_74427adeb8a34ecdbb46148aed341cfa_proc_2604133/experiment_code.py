import os, random, string, datetime, json
from typing import List, Dict, Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------- working dir & logging dict --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# ---------------- device ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- dataset loading / synthetic fallback ------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(p: str) -> bool:
    return all(
        os.path.isfile(os.path.join(p, f"{sp}.csv")) for sp in ["train", "dev", "test"]
    )


if not spr_files_exist(SPR_PATH):
    print("SPR_BENCH not found, building synthetic data for demo.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = [str(i) for i in range(4)]  # 0-3

    def rand_seq() -> str:
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(seq: str) -> int:
        return int(
            len(set(t[0] for t in seq.split())) == len(set(t[1] for t in seq.split()))
        )

    def make_split(n: int) -> Dict[str, List]:
        seqs = [rand_seq() for _ in range(n)]
        return {"sequence": seqs, "label": [rule(s) for s in seqs]}

    raw = {"train": make_split(2000), "dev": make_split(400), "test": make_split(600)}
else:
    print("Loading real SPR_BENCH")
    from datasets import load_dataset, DatasetDict

    def load_spr(root: str) -> DatasetDict:
        def _load(fname: str):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, fname),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict()
        for sp in ["train", "dev", "test"]:
            d[sp] = _load(f"{sp}.csv")
        return d

    dset = load_spr(SPR_PATH)
    raw = {
        sp: {"sequence": dset[sp]["sequence"], "label": dset[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }

# ---------------- helper functions -------------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    tokset = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(tokset))})
    return vocab


vocab = build_vocab(raw["train"]["sequence"])
print(f"Vocabulary size = {len(vocab)}")


def encode(seq: str) -> List[int]:
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


def sym_feats(seq: str) -> Tuple[int, int, int]:
    sv = len(set(t[0] for t in seq.split()))
    cv = len(set(t[1] for t in seq.split()))
    return sv, cv, int(sv == cv)


def shape_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [len(set(t[0] for t in s.split())) for s in seqs]
    return sum(w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)) / (
        sum(weights) or 1
    )


# ---------------- torch Dataset & DataLoader ---------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs: List[str], labels: List[int]):
        self.raw_seq = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        s = self.raw_seq[idx]
        return {
            "input_ids": torch.tensor(encode(s), dtype=torch.long),
            "sym": torch.tensor(sym_feats(s), dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    maxlen = max(lengths)
    ids = torch.full((len(batch), maxlen), fill_value=vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    sym = torch.stack([b["sym"] for b in batch])
    return {
        "input_ids": ids,
        "labels": labels,
        "lengths": torch.tensor(lengths),
        "sym": sym,
    }


datasets = {
    sp: SPRDataset(raw[sp]["sequence"], raw[sp]["label"])
    for sp in ["train", "dev", "test"]
}
loaders = {
    sp: DataLoader(
        datasets[sp], batch_size=64, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in datasets
}


# ---------------- neuro-symbolic model ---------------------------------
class NeuroSymbolic(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hid: int = 128,
        sym_dim: int = 3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.rnn = nn.GRU(embed_dim, hid, batch_first=True)
        self.sym_proj = nn.Sequential(nn.Linear(sym_dim, hid // 2), nn.ReLU())
        self.fc = nn.Linear(hid + hid // 2, num_classes)

    def forward(self, ids, lengths, sym):
        e = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)  # (1,B,H)
        h = h.squeeze(0)  # (B,H)
        s = self.sym_proj(sym)
        cat = torch.cat([h, s], dim=1)
        return self.fc(cat)


num_classes = len(set(raw["train"]["label"]))
model = NeuroSymbolic(len(vocab), 64, 128, 3, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------- evaluation ------------------------------------------
@torch.no_grad()
def run_eval(split: str, need_preds: bool = False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds = []
    for batch in loaders[split]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["lengths"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
        if need_preds:
            all_preds.extend(preds.cpu().tolist())
    acc = correct / total
    if not need_preds:
        return acc, loss_sum / total
    else:
        return acc, loss_sum / total, all_preds


# ---------------- training with early stopping ------------------------
best_val_loss = float("inf")
patience = 3
wait = 0
best_state = None
max_epochs = 20

for epoch in range(1, max_epochs + 1):
    model.train()
    running = 0.0
    for batch in loaders["train"]:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["lengths"], batch["sym"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["labels"].size(0)
    train_loss = running / len(datasets["train"])
    train_acc, _ = run_eval("train")
    val_acc, val_loss = run_eval("dev")

    ed = experiment_data["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc})
    ed["metrics"]["val"].append({"acc": val_acc})
    ed["timestamps"].append(str(datetime.datetime.now()))
    print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---------------- final evaluation ------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

test_acc, test_loss, test_preds = run_eval("test", need_preds=True)
test_swa = shape_weighted_accuracy(
    datasets["test"].raw_seq, datasets["test"].labels, test_preds
)
print(f"TEST: acc={test_acc:.3f}  SWA={test_swa:.3f}")

ed = experiment_data["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": test_acc, "swa": test_swa}
ed["predictions"] = test_preds
ed["ground_truth"] = datasets["test"].labels

# ---------------- save logs & plot ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

plt.figure()
plt.plot(ed["losses"]["train"], label="train")
plt.plot(ed["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()
