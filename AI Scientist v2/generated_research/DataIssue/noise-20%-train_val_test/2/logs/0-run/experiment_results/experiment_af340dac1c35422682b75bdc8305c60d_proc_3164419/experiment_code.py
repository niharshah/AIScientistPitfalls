# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- working dir --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
experiment_data = {
    "pretrain+cls": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- dataset utilities --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


class SPRSeqDataset(Dataset):
    """For classification."""

    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.max_len, self.pad = vocab, max_len, vocab["<pad>"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = [
            self.vocab.get(ch, self.vocab["<unk>"])
            for ch in self.seqs[idx][: self.max_len]
        ]
        seq += [self.pad] * (self.max_len - len(seq))
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CausalLMDataset(Dataset):
    """For next-token prediction pre-training."""

    def __init__(self, hf_ds, vocab, max_len):
        self.seqs = hf_ds["sequence"]
        self.vocab, self.max_len, self.pad = vocab, max_len, vocab["<pad>"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_ids = [
            self.vocab.get(ch, self.vocab["<unk>"])
            for ch in self.seqs[idx][: self.max_len - 1]
        ]
        inp = [self.pad] + seq_ids  # shift right
        tgt = seq_ids + [self.pad]  # predict each original char incl. last pad
        inp += [self.pad] * (self.max_len - len(inp))
        tgt += [self.pad] * (self.max_len - len(tgt))
        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(tgt, dtype=torch.long),
        }


# -------------------- model definitions --------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, causal=False):
        # x : (B,L)
        h = self.embed(x) + self.pos[:, : x.size(1)]
        if causal:
            L = x.size(1)
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h.transpose(0, 1), mask).transpose(0, 1)
        else:
            h = self.enc(h.transpose(0, 1)).transpose(0, 1)
        return h


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.encoder(x, causal=True)
        return self.lm_head(h)


class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers, max_len)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.encoder(x).mean(1)
        return self.cls_head(h)


# -------------------- training helpers --------------------
def train_causal_epoch(model, loader, optim, criterion):
    model.train()
    tot = 0
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        loss.backward()
        optim.step()
        loss_sum += loss.item() * batch["labels"].size(0)
        tot += batch["labels"].size(0)
    return loss_sum / tot


def train_cls_epoch(model, loader, optim, criterion):
    model.train()
    tot_loss = 0
    preds = []
    gts = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optim.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    return tot_loss / len(loader.dataset), f1_score(gts, preds, average="macro")


@torch.no_grad()
def eval_cls_epoch(model, loader, criterion):
    model.eval()
    tot_loss = 0
    preds = []
    gts = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), f1, preds, gts


# -------------------- main execution --------------------
def run():
    # ---- load data ----
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    dsets = load_spr_bench(DATA_PATH)

    # ---- vocab ----
    chars = set("".join(dsets["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({c: i + 2 for i, c in enumerate(sorted(chars))})
    max_len = min(128, max(len(s) for s in dsets["train"]["sequence"]) + 1)
    print("Vocab size:", len(vocab), "Max_len:", max_len)

    # ---- dataloaders ----
    pretrain_loader = DataLoader(
        CausalLMDataset(dsets["train"], vocab, max_len),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        SPRSeqDataset(dsets["train"], vocab, max_len), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(SPRSeqDataset(dsets["dev"], vocab, max_len), batch_size=256)
    test_loader = DataLoader(
        SPRSeqDataset(dsets["test"], vocab, max_len), batch_size=256
    )

    # ---- model configs ----
    d_model, nhead, nlayer = 128, 4, 3
    vocab_size = len(vocab)
    num_classes = len(set(dsets["train"]["label"]))

    # ========== Stage 1: Causal LM pre-training ==========
    lm = CausalLM(vocab_size, d_model, nhead, nlayer, max_len).to(device)
    opt_lm = torch.optim.Adam(lm.parameters(), lr=1e-3)
    crit_lm = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    pre_epochs = 3
    for ep in range(1, pre_epochs + 1):
        l = train_causal_epoch(lm, pretrain_loader, opt_lm, crit_lm)
        experiment_data["pretrain+cls"]["losses"]["pretrain"].append(l)
        print(f"Pre-train epoch {ep}/{pre_epochs}: loss={l:.4f}")

    # save encoder weights
    enc_state = lm.encoder.state_dict()

    # ========== Stage 2: Classification fine-tuning ==========
    cls_model = SPRClassifier(
        vocab_size, num_classes, d_model, nhead, nlayer, max_len
    ).to(device)
    cls_model.encoder.load_state_dict(enc_state)  # transfer

    optim_cls = torch.optim.Adam(cls_model.parameters(), lr=5e-4)
    crit_cls = nn.CrossEntropyLoss()

    best_f1 = 0.0
    wait = 0
    patience = 5
    best_state = None
    max_epochs = 15
    for ep in range(1, max_epochs + 1):
        tr_loss, tr_f1 = train_cls_epoch(cls_model, train_loader, optim_cls, crit_cls)
        val_loss, val_f1, _, _ = eval_cls_epoch(cls_model, val_loader, crit_cls)

        ed = experiment_data["pretrain+cls"]
        ed["epochs"].append(ep)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_macro_f1"].append(tr_f1)
        ed["metrics"]["val_macro_f1"].append(val_f1)

        print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = cls_model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

    # ---- test evaluation ----
    if best_state:
        cls_model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = eval_cls_epoch(cls_model, test_loader, crit_cls)
    print(f"TEST macro-F1 = {test_f1:.4f}")
    ed = experiment_data["pretrain+cls"]
    ed["predictions"], ed["ground_truth"] = preds, gts
    ed["test_macro_f1"], ed["test_loss"] = test_f1, test_loss

    # ---- save ----
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


run()
