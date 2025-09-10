import os, pathlib, math, time, random, json
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score

# ---- working dir & device ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- experiment storage ----
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None, "SGA": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---- dataset --------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):  # treat each csv as a single split
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

# ---- vocabulary & helpers --------------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({c for s in spr["train"]["sequence"] for c in s})
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
vocab_size, num_classes = len(vocab), len(label2id)
MAX_LEN = 64
print(f"vocab_size={vocab_size}  num_classes={num_classes}")


def encode_seq(seq: str):
    ids = [stoi.get(c, stoi[UNK]) for c in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def sym_count(seq: str):
    v = np.zeros(vocab_size, dtype=np.float32)
    for c in seq:
        v[stoi.get(c, stoi[UNK])] += 1
    return v / max(1, len(seq))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labs = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "sym_counts": torch.tensor(sym_count(self.seqs[idx]), dtype=torch.float32),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# ---- model -----------------------------------------------------------------
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridMultiTask(nn.Module):
    def __init__(self, vocab, n_classes, d_model=128, nhead=4, nlayers=4, drop=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, drop, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.sym_fc = nn.Linear(vocab, d_model)
        self.cls_head = nn.Linear(d_model * 2, n_classes)  # classification
        self.next_head = nn.Linear(d_model, vocab)  # next-token

    def forward(self, input_ids, sym_counts):
        mask = input_ids.eq(0)
        x = self.pos(self.embed(input_ids))
        x = self.enc(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0)
        seq_repr = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        sym_repr = torch.relu(self.sym_fc(sym_counts))
        cls_logits = self.cls_head(torch.cat([seq_repr, sym_repr], dim=-1))
        next_logits = self.next_head(x)  # per-position logits
        return cls_logits, next_logits


model = HybridMultiTask(vocab_size, num_classes).to(device)

# ---- optim & loss ----------------------------------------------------------
criterion_cls = nn.CrossEntropyLoss()
criterion_next = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


# ---- train / eval loops ----------------------------------------------------
def prep_batch(batch):
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, preds_all, labs_all = 0.0, [], []
    for batch in loader:
        batch = prep_batch(batch)
        with torch.set_grad_enabled(train):
            cls_logits, nxt_logits = model(batch["input_ids"], batch["sym_counts"])
            # next-token targets (shift left)
            tgt = batch["input_ids"][:, 1:].contiguous()
            pad = torch.full((tgt.size(0), 1), -100, dtype=torch.long, device=device)
            tgt = torch.cat([tgt, pad], dim=1)  # same length as input
            loss_cls = criterion_cls(cls_logits, batch["labels"])
            loss_next = criterion_next(nxt_logits.view(-1, vocab_size), tgt.view(-1))
            loss = loss_cls + 0.5 * loss_next
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds_all.append(cls_logits.argmax(-1).cpu())
        labs_all.append(batch["labels"].cpu())
    preds = torch.cat(preds_all).numpy()
    labs = torch.cat(labs_all).numpy()
    f1 = f1_score(labs, preds, average="macro")
    return total_loss / len(loader.dataset), f1, preds, labs


# ---- training loop with early stopping ------------------------------------
EPOCHS, best_val, patience, wait = 20, -1, 4, 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(vl_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(f"Epoch {epoch}: val_loss={vl_loss:.4f}  val_F1={vl_f1:.4f}")
    if vl_f1 > best_val:
        best_val = vl_f1
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_state)

# ---- evaluation on test ----------------------------------------------------
ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(test_dl)
print(f"Test Macro-F1 = {ts_f1:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test_f1"] = float(ts_f1)
experiment_data["SPR_BENCH"]["predictions"] = ts_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = ts_labels.tolist()


# ---- SGA proxy: unseen bigram combinations ---------------------------------
def bigrams(seq):
    return {seq[i : i + 2] for i in range(len(seq) - 1)}


train_bigrams = set().union(*(bigrams(s) for s in spr["train"]["sequence"]))
ood_mask = np.array(
    [len(bigrams(s) - train_bigrams) > 0 for s in spr["test"]["sequence"]]
)
correct = ts_preds == ts_labels
SGA = correct[ood_mask].mean() if ood_mask.any() else 0.0
print(f"Systematic Generalization Accuracy (proxy) = {SGA:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["SGA"] = float(SGA)

# ---- save everything -------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
