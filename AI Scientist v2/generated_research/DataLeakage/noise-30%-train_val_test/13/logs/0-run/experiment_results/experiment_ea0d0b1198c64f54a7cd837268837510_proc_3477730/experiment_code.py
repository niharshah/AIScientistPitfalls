import os, pathlib, math, time, random, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# -------------------------------------------------- paths / dirs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------- device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------- experiment log skeleton
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None, "SGA": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
exp_rec = experiment_data["SPR_BENCH"]


# -------------------------------------------------- dataset helpers
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# find dataset folder
for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found")

spr = load_spr_bench(DATA_PATH)

# -------------------------------------------------- vocab & labels
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(
    {c for s in spr["train"]["sequence"] for c in s}
)
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}

vocab_size = len(vocab)
num_classes = len(label2id)
MAX_LEN = max(len(seq) for split in spr for seq in spr[split]["sequence"])  # dynamic!
print(f"vocab_size={vocab_size} | num_classes={num_classes} | MAX_LEN={MAX_LEN}")


# -------------------------------------------------- dataset class
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        token_ids = [stoi.get(c, stoi[UNK_TOKEN]) for c in seq]  # no truncation
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


def collate_batch(batch):
    ids = [b["input_ids"] for b in batch]
    labs = torch.stack([b["labels"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(
        ids, batch_first=True, padding_value=stoi[PAD_TOKEN]
    )
    return {"input_ids": padded.to(device), "labels": labs.to(device)}


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)
val_dl = DataLoader(
    SPRTorchDataset(spr["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)
test_dl = DataLoader(
    SPRTorchDataset(spr["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)


# -------------------------------------------------- model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_sz, d_model, nhead, nlayers, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, d_model, padding_idx=stoi[PAD_TOKEN])
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, input_ids):
        mask = input_ids.eq(stoi[PAD_TOKEN])
        x = self.pos(self.embed(input_ids))
        x = self.enc(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0)
        seq_repr = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.out(seq_repr)


model = TransformerClassifier(vocab_size, 128, 4, 2, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------------------------------------- training / evaluation helpers
def run_epoch(dataloader, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, labs = 0.0, [], []
    for batch in dataloader:
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).detach().cpu())
        labs.append(batch["labels"].detach().cpu())
    preds = torch.cat(preds).numpy()
    labs = torch.cat(labs).numpy()
    return (
        tot_loss / len(dataloader.dataset),
        f1_score(labs, preds, average="macro"),
        preds,
        labs,
    )


# -------------------------------------------------- training loop
EPOCHS, best_val, patience, wait = 15, -1, 3, 0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, train=True)
    vl_loss, vl_f1, _, _ = run_epoch(val_dl, train=False)

    exp_rec["losses"]["train"].append(tr_loss)
    exp_rec["losses"]["val"].append(vl_loss)
    exp_rec["metrics"]["train_f1"].append(tr_f1)
    exp_rec["metrics"]["val_f1"].append(vl_f1)
    exp_rec["epochs"].append(epoch)

    print(f"Epoch {epoch}: val_loss = {vl_loss:.4f} | val_macroF1 = {vl_f1:.4f}")

    if vl_f1 > best_val:
        best_val = vl_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)

# -------------------------------------------------- test evaluation
ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(test_dl)
print(f"Test macro-F1 = {ts_f1:.4f}")
exp_rec["metrics"]["test_f1"] = float(ts_f1)
exp_rec["predictions"] = ts_preds.tolist()
exp_rec["ground_truth"] = ts_labels.tolist()


# -------------------------------------------------- SGA proxy
def bigrams(seq):
    return {seq[i : i + 2] for i in range(len(seq) - 1)}


train_bigrams = set().union(*(bigrams(s) for s in spr["train"]["sequence"]))
ood_mask = np.array(
    [len(bigrams(s) - train_bigrams) > 0 for s in spr["test"]["sequence"]]
)
correct = ts_preds == ts_labels
SGA = correct[ood_mask].mean() if ood_mask.any() else 0.0
print(f"Systematic Generalization Accuracy (proxy) = {SGA:.4f}")
exp_rec["metrics"]["SGA"] = float(SGA)

# -------------------------------------------------- save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
