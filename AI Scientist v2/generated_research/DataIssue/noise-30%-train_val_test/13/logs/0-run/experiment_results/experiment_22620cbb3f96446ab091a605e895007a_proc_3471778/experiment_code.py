import os, pathlib, math, collections, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------ housekeeping & device ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None, "SGA": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------ SPR loading utility ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


for cand in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if cand.exists():
        DATA_PATH = cand
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found!")

spr = load_spr_bench(DATA_PATH)

# ------------------ vocab & helpers ------------------
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


# ------------------ bigram signature ------------------
TOP_N = 512
bigram_counter = collections.Counter()
for seq in spr["train"]["sequence"]:
    bigram_counter.update(seq[i : i + 2] for i in range(len(seq) - 1))
top_bigrams = [bg for bg, _ in bigram_counter.most_common(TOP_N)]
bigram2idx = {bg: i for i, bg in enumerate(top_bigrams)}
BIGRAM_DIM = TOP_N


def bigram_vector(seq):
    vec = np.zeros(BIGRAM_DIM, dtype=np.float32)
    for i in range(len(seq) - 1):
        bg = seq[i : i + 2]
        j = bigram2idx.get(bg)
        if j is not None:
            vec[j] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()  # normalise frequency
    return vec


# ------------------ Torch dataset ------------------
class SPRTorchDS(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labs = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode_seq(seq), dtype=torch.long),
            "bigrams": torch.tensor(bigram_vector(seq), dtype=torch.float32),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(SPRTorchDS(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRTorchDS(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorchDS(spr["test"]), batch_size=batch_size)


# ------------------ model ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class RelationalHybrid(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, bigram_dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.trans = nn.TransformerEncoder(enc_layer, nlayers)
        self.bgram_mlp = nn.Sequential(
            nn.Linear(bigram_dim, 128), nn.ReLU(), nn.Linear(128, d_model), nn.ReLU()
        )
        self.fc_out = nn.Linear(d_model * 2, n_classes)

    def forward(self, input_ids, bigrams):
        mask = input_ids.eq(0)
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.trans(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0.0)
        seq_repr = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        rel_repr = self.bgram_mlp(bigrams)
        combined = torch.cat([seq_repr, rel_repr], dim=-1)
        return self.fc_out(combined)


model = RelationalHybrid(vocab_size, 128, 4, 2, BIGRAM_DIM, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------ training utils ------------------
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, labs = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["bigrams"])
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


# ------------------ train loop ------------------
EPOCHS, best_val = 20, -1
patience, wait = 4, 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_dl)
    print(f"Epoch {epoch}: validation_loss = {vl_loss:.4f} | val_F1 = {vl_f1:.4f}")
    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(vl_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    # early stopping
    if vl_f1 > best_val:
        best_val = vl_f1
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)

# ------------------ evaluation ------------------
ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(test_dl)
print(f"Test macro-F1 = {ts_f1:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test_f1"] = ts_f1
experiment_data["SPR_BENCH"]["predictions"] = ts_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = ts_labels.tolist()


# ---------- proxy Systematic Generalization Accuracy ----------
def bigrams(seq):
    return {seq[i : i + 2] for i in range(len(seq) - 1)}


train_bigrams_set = set().union(*(bigrams(s) for s in spr["train"]["sequence"]))
ood_mask = np.array(
    [len(bigrams(s) - train_bigrams_set) > 0 for s in spr["test"]["sequence"]]
)
correct = ts_preds == ts_labels
SGA = correct[ood_mask].mean() if ood_mask.any() else 0.0
print(f"Systematic Generalization Accuracy (proxy) = {SGA:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["SGA"] = float(SGA)

# ------------------ save ------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
