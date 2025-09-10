import os, pathlib, numpy as np, torch, math, time, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------#
# working dir / device --------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment data container ---------------------------------------------------#
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------------------------#
# utility: locate dataset -----------------------------------------------------#
def _find_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path(os.getenv("SPR_DATASET_PATH", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]
    req = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and req.issubset({p.name for p in c.iterdir()}):
            print(f"Found SPR_BENCH at {c}")
            return c
    raise FileNotFoundError("SPR_BENCH folder not found; set env vars if necessary.")


# -----------------------------------------------------------------------------#
# load dataset ----------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {"train": _load("train"), "dev": _load("dev"), "test": _load("test")}
    )


# -----------------------------------------------------------------------------#
# vocab / dataset classes -----------------------------------------------------#
class SPRCharDataset(Dataset):
    def __init__(self, hf_split, vocab, max_len):
        self.data = hf_split
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.cls_id = vocab["<cls>"]
        self.max_len = max_len

    def _encode(self, seq: str):
        seq = seq.replace(" ", "")
        ids = [self.cls_id] + [self.vocab[ch] for ch in seq]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = self._encode(row["sequence"])
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        parity = torch.tensor(
            len(row["sequence"].replace(" ", "")) % 2, dtype=torch.long
        )
        return {"input_ids": ids, "labels": label, "parity": parity}


def build_vocab(train_split):
    chars = {ch for ex in train_split for ch in ex["sequence"].replace(" ", "")}
    vocab = {"<pad>": 0, "<cls>": 1}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


def collate_fn(batch, pad_id):
    keys = ["input_ids", "labels", "parity"]
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    parity = torch.stack([b["parity"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    mask = (padded != pad_id).long()
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "labels": labels,
        "parity": parity,
    }


# -----------------------------------------------------------------------------#
# model -----------------------------------------------------------------------#
class TransformerWithAux(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        max_len,
        d_model=128,
        nhead=4,
        nlayers=3,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(max_len + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls_head = nn.Linear(d_model, num_labels)
        self.parity_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.size()
        x = self.embed(input_ids) + self.pos[:L]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        cls_tok = x[:, 0, :]  # representation of [CLS]
        logits_main = self.cls_head(cls_tok)
        logits_parity = self.parity_head(cls_tok)
        return logits_main, logits_parity


# -----------------------------------------------------------------------------#
# train / eval loops -----------------------------------------------------------#
def run_epoch(model, loader, crit_main, crit_aux, aux_lambda, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = tot_acc = tot_size = 0.0
    all_preds, all_gts = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            logits_main, logits_aux = model(batch["input_ids"], batch["attention_mask"])
            loss_main = crit_main(logits_main, batch["labels"])
            loss_aux = crit_aux(logits_aux, batch["parity"])
            loss = loss_main + aux_lambda * loss_aux
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits_main.argmax(1)
            tot_acc += (preds == batch["labels"]).sum().item()
            tot_size += batch["labels"].size(0)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(batch["labels"].cpu().numpy())
    acc = tot_acc / tot_size
    f1 = f1_score(all_gts, all_preds, average="macro")
    return tot_loss / tot_size, acc, f1


# -----------------------------------------------------------------------------#
# pipeline --------------------------------------------------------------------#
data_dir = _find_spr_bench()
dset = load_spr_bench(data_dir)
num_labels = len(set(int(ex["label"]) for ex in dset["train"]))
vocab = build_vocab(dset["train"])
max_len = (
    max(len(ex["sequence"].replace(" ", "")) for ex in dset["train"]) + 1
)  # +1 for CLS
print(f"Vocab size {len(vocab)}, max_len {max_len}, num_labels {num_labels}")

train_ds = SPRCharDataset(dset["train"], vocab, max_len)
dev_ds = SPRCharDataset(dset["dev"], vocab, max_len)
test_ds = SPRCharDataset(dset["test"], vocab, max_len)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)

model = TransformerWithAux(len(vocab), num_labels, max_len, dropout=0.1).to(device)
try:
    crit_main = nn.CrossEntropyLoss(label_smoothing=0.1)
except TypeError:  # fallback if older torch
    crit_main = nn.CrossEntropyLoss()
crit_aux = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
aux_lambda = 0.3
epochs = 8
best_val_f1 = -1.0
best_state = None

for epoch in range(1, epochs + 1):
    t_loss, t_acc, t_f1 = run_epoch(
        model, train_loader, crit_main, crit_aux, aux_lambda, optimizer
    )
    v_loss, v_acc, v_f1 = run_epoch(model, dev_loader, crit_main, crit_aux, aux_lambda)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append({"acc": t_acc, "f1": t_f1})
    experiment_data["SPR_BENCH"]["metrics"]["val"].append({"acc": v_acc, "f1": v_f1})
    experiment_data["SPR_BENCH"]["losses"]["train"].append(t_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    print(
        f"Epoch {epoch}: validation_loss = {v_loss:.4f} | Val Acc {v_acc*100:.2f}% | Val Macro-F1 {v_f1:.4f}"
    )
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        best_state = model.state_dict()

# -----------------------------------------------------------------------------#
# evaluate on test ------------------------------------------------------------#
model.load_state_dict(best_state)
test_loss, test_acc, test_f1 = run_epoch(
    model, test_loader, crit_main, crit_aux, aux_lambda
)
print(
    f"TEST  ->  loss {test_loss:.4f} | Acc {test_acc*100:.2f}% | Macro-F1 {test_f1:.4f}"
)

# collect predictions / ground truth for test set
model.eval()
preds_all, gts_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
        preds_all.extend(logits.argmax(1).cpu().numpy())
        gts_all.extend(batch["labels"].cpu().numpy())
experiment_data["SPR_BENCH"]["predictions"] = preds_all
experiment_data["SPR_BENCH"]["ground_truth"] = gts_all

# save everything -------------------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}")
