# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, time, random, math, json, re, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------------
# 0) Reproducibility & device
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------------
# 1) Working dir & experiment data dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"d_model_tuning": {}}  # each key will be a d_model value as string


# -------------------------------------------------------------------------------
# 2) Load SPR benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    dd = DatasetDict()
    dd["train"] = _load("train.csv")
    dd["dev"] = _load("dev.csv")
    dd["test"] = _load("test.csv")
    return dd


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------------------
# 3) Build vocab (whitespace)
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add_tok(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add_tok(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)

# -------------------------------------------------------------------------------
# 4) Dataset & dataloaders
MAX_LEN = 128


def encode_sequence(seq: str):
    toks = [CLS] + seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad_len = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad_len
        attn += [0] * pad_len
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


train_ds, dev_ds = SPRDataset(spr["train"]), SPRDataset(spr["dev"])
BATCH_SIZE = 64


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# -------------------------------------------------------------------------------
# 5) Transformer model
class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=128, nhead=4, nlayers=2, dim_ff=256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        logits = self.classifier(x[:, 0, :])
        return logits


# -------------------------------------------------------------------------------
# 6) Train / eval helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# -------------------------------------------------------------------------------
# 7) Hyperparameter grid search over d_model
DMODEL_GRID = [64, 128, 256, 384]


def pick_head(d):
    if d % 8 == 0:
        return 8
    if d % 4 == 0:
        return 4
    return 1


EPOCHS = 5

for d_model in DMODEL_GRID:
    run_key = str(d_model)
    experiment_data["d_model_tuning"][run_key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    print(f"\n=== Training with d_model={d_model} ===")
    model = SimpleTransformer(
        vocab_size,
        num_labels,
        d_model=d_model,
        nhead=pick_head(d_model),
        nlayers=2,
        dim_ff=d_model * 2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_f1, val_pred, val_gt = run_epoch(model, dev_loader)

        exp_dict = experiment_data["d_model_tuning"][run_key]["SPR_BENCH"]
        exp_dict["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1})
        exp_dict["metrics"]["val"].append({"epoch": epoch, "macro_f1": val_f1})
        exp_dict["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        exp_dict["losses"]["val"].append({"epoch": epoch, "loss": val_loss})

        print(
            f"d_model={d_model} | Epoch {epoch}: "
            f"train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_F1={tr_f1:.4f}, val_F1={val_f1:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # store last predictions
    exp_dict["predictions"] = val_pred
    exp_dict["ground_truth"] = val_gt

# -------------------------------------------------------------------------------
# 8) Save all results
save_path = os.path.join(working_dir, "experiment_data.npy")
np.save(save_path, experiment_data, allow_pickle=True)
print("Saved experiment data to", save_path)
