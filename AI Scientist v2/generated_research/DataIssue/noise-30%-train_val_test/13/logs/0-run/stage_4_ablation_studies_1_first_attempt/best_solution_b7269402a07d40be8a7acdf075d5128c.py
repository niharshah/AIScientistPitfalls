# BiGRU-Sequence-Encoder ablation for SPR-BENCH
import os, pathlib, math, time, json, random
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score

# ---------- dirs / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "BiGRU": {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": [], "test_f1": None, "SGA": None},
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
itos = {i: c for c, i in stoi.items()}
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
vocab_size, num_classes = len(vocab), len(label2id)
MAX_LEN = 64
print(f"vocab_size={vocab_size}, num_classes={num_classes}")


def encode_seq(seq):
    ids = [stoi.get(c, stoi[UNK]) for c in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def bag_of_symbols(seq):
    v = np.zeros(vocab_size, dtype=np.float32)
    for c in seq:
        v[stoi.get(c, 1)] += 1
    return v / max(1, len(seq))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labs = hf_split["sequence"], hf_split["label"]

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


# ---------- model ----------
class BiGRUEncoder(nn.Module):
    """Two-layer bidirectional GRU returning sequence outputs of dim d_model."""

    def __init__(self, vocab, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.bigru = nn.GRU(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

    def forward(self, input_ids):
        # input_ids: [B, L]
        mask = input_ids.eq(0)  # PAD mask
        x = self.embed(input_ids)  # [B, L, d_model]
        lengths = (~mask).sum(1).cpu()
        # pack to avoid useless computation
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.bigru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=MAX_LEN
        )
        out.masked_fill_(mask.unsqueeze(-1), 0)
        return out, mask


class HybridGRUClassifier(nn.Module):
    def __init__(self, vocab, d_model, sym_dim, n_classes):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab, d_model)
        self.sym_fc = nn.Linear(sym_dim, d_model)
        self.out = nn.Linear(d_model * 2, n_classes)

    def forward(self, input_ids, sym_counts):
        seq_out, mask = self.encoder(input_ids)  # [B,L,d_model]
        seq_repr = seq_out.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        sym_repr = torch.relu(self.sym_fc(sym_counts))
        combined = torch.cat([seq_repr, sym_repr], dim=-1)
        return self.out(combined)


d_model = 128
model = HybridGRUClassifier(vocab_size, d_model, vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- helper ----------
def run_epoch(model, dl, train=False):
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
EPOCHS, best_val = 15, -1
patience, wait = 3, 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(model, val_dl)
    ed = experiment_data["BiGRU"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(vl_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(vl_f1)
    ed["epochs"].append(epoch)
    print(f"Epoch {epoch}: val_loss={vl_loss:.4f} val_F1={vl_f1:.4f}")
    if vl_f1 > best_val:
        best_val = vl_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stop")
            break

model.load_state_dict(best_state)

# ---------- test ----------
ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(model, test_dl)
print(f"Test macro-F1={ts_f1:.4f}")
ed = experiment_data["BiGRU"]["SPR_BENCH"]
ed["metrics"]["test_f1"] = ts_f1
ed["predictions"] = ts_preds.tolist()
ed["ground_truth"] = ts_labels.tolist()


# ---------- SGA metric (OOD bigram proxy) ----------
def bigrams(seq):
    return {seq[i : i + 2] for i in range(len(seq) - 1)}


train_bigrams = set().union(*(bigrams(s) for s in spr["train"]["sequence"]))
ood_mask = []
for s in spr["test"]["sequence"]:
    ood_mask.append(len(bigrams(s) - train_bigrams) > 0)
ood_mask = np.array(ood_mask)
correct = ts_preds == ts_labels
SGA = correct[ood_mask].mean() if ood_mask.any() else 0.0
print(f"Systematic Generalization Accuracy (proxy) = {SGA:.4f}")
ed["metrics"]["SGA"] = float(SGA)

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
