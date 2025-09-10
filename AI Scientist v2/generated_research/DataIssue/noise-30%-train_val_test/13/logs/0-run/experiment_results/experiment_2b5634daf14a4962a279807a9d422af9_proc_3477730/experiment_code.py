import os, pathlib, math, time, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------- reproducibility ----------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------- dirs / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment dict ----------
experiment_data = {
    "no_positional_encoding": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": None, "SGA": None},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH not found")
spr = load_spr_bench(DATA_PATH)

# ---------- vocab / labels ----------
PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({c for s in spr["train"]["sequence"] for c in s})
stoi = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
num_classes = len(label2id)
MAX_LEN = 64
print(f"vocab_size={vocab_size}, num_classes={num_classes}")


def encode_seq(seq):
    ids = [stoi.get(c, stoi[UNK]) for c in seq[:MAX_LEN]]
    return ids + [stoi[PAD]] * (MAX_LEN - len(ids))


def bag_of_symbols(seq):
    v = np.zeros(vocab_size, dtype=np.float32)
    for c in seq:
        v[stoi.get(c, 1)] += 1
    return v / max(1, len(seq))


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labs = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "sym_counts": torch.tensor(
                bag_of_symbols(self.seqs[idx]), dtype=torch.float32
            ),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=batch_size)


# ---------- model (NO positional encoding) ----------
class HybridClassifierNoPos(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, sym_dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_classes)

    def forward(self, input_ids, sym_counts):
        mask = input_ids.eq(0)
        x = self.embed(input_ids)  # NO positional encoding added
        x = self.transformer(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0)
        seq_repr = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        sym_repr = torch.relu(self.sym_fc(sym_counts))
        return self.out(torch.cat([seq_repr, sym_repr], -1))


model = HybridClassifierNoPos(vocab_size, 128, 4, 2, vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- helper ----------
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, labs = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["sym_counts"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labs.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labs = torch.cat(labs).numpy()
    return (
        total_loss / len(dl.dataset),
        f1_score(labs, preds, average="macro"),
        preds,
        labs,
    )


# ---------- training ----------
EPOCHS, best_val, patience, wait = 15, -1, 3, 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_dl)
    ed = experiment_data["no_positional_encoding"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(vl_loss)
    ed["metrics"]["train"].append(tr_f1)
    ed["metrics"]["val"].append(vl_f1)
    ed["epochs"].append(epoch)
    print(f"Epoch {epoch}: val_loss={vl_loss:.4f} val_F1={vl_f1:.4f}")
    if vl_f1 > best_val:
        best_val, best_state, wait = vl_f1, model.state_dict(), 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break
model.load_state_dict(best_state)

# ---------- test ----------
ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(test_dl)
print(f"Test macro-F1={ts_f1:.4f}")
ed = experiment_data["no_positional_encoding"]["SPR_BENCH"]
ed["metrics"]["test"] = ts_f1
ed["predictions"] = ts_preds.tolist()
ed["ground_truth"] = ts_labels.tolist()


# ---------- SGA ----------
def bigrams(seq):
    return {seq[i : i + 2] for i in range(len(seq) - 1)}


train_bigrams = set().union(*(bigrams(s) for s in spr["train"]["sequence"]))
ood_mask = np.array(
    [len(bigrams(s) - train_bigrams) > 0 for s in spr["test"]["sequence"]]
)
correct = ts_preds == ts_labels
SGA = correct[ood_mask].mean() if ood_mask.any() else 0.0
print(f"Systematic Generalization Accuracy (proxy) = {SGA:.4f}")
ed["metrics"]["SGA"] = float(SGA)

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
