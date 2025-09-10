# nhead-tuning for SPR-BENCH ------------------------------------------------------------
import os, math, pathlib, random, time, gc
from typing import List, Dict

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict


# ----------------------------------------------------------------------------- misc ---
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# ----------------------------------------------------------------- experiment dict ----
experiment_data = {
    "nhead_tuning": {
        "SPR_BENCH": {
            "nhead_values": [],
            "metrics": {
                "train_macro_f1": [],
                "val_macro_f1": [],
                "test_macro_f1": [],
            },
            "losses": {"train": [], "val": [], "test": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# -------------------------------------------------------------------------- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------ data loading / vocab ---
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split if split != "dev" else "dev"] = _load(f"{split}.csv")
    return dset


data_path = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adapt if needed
spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
label_set = sorted(set(spr["train"]["label"]))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)


def encode(seq: str) -> List[int]:
    tok_ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()][:max_len]
    tok_ids += [vocab[PAD]] * (max_len - len(tok_ids))
    return tok_ids


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader_full = DataLoader(
    SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_loader_full = DataLoader(
    SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False
)
test_loader_full = DataLoader(
    SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False
)


# ------------------------------------------------------------------- model pieces ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(
            position * div_term
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_sz: int,
        emb_dim: int,
        nhead: int,
        num_layers: int,
        n_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim,
            nhead,
            dim_feedforward=4 * emb_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(emb_dim, n_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids == 0
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.classifier(x)


# ---------------------------------------------------------------- training helpers ---
def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        all_preds.extend(logits.argmax(-1).detach().cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    avg_loss = tot_loss / len(loader.dataset)
    return (
        avg_loss,
        f1_score(all_trues, all_preds, average="macro"),
        all_preds,
        all_trues,
    )


# ------------------------------------------------------------- hyper-parameter loop ---
nhead_grid = [4, 8, 16]  # emb_dim (128) divisible by all
num_epochs = 8

for nhead in nhead_grid:
    print(f"\n===== Training with nhead = {nhead} =====")
    experiment_data["nhead_tuning"]["SPR_BENCH"]["nhead_values"].append(nhead)

    # fresh dataloaders (to reset shuffling each run)
    train_loader = train_loader_full
    val_loader = val_loader_full
    test_loader = test_loader_full

    model = CharTransformer(
        vocab_size, 128, nhead, num_layers=2, n_labels=num_labels
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cfg_train_losses, cfg_val_losses = [], []
    cfg_train_f1s, cfg_val_f1s = [], []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)
        cfg_train_losses.append(tr_loss)
        cfg_val_losses.append(val_loss)
        cfg_train_f1s.append(tr_f1)
        cfg_val_f1s.append(val_f1)
        print(
            f"[nhead={nhead}] Epoch {epoch}/{num_epochs} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} ({time.time() - t0:.1f}s)"
        )

    # final test
    test_loss, test_f1, test_preds, test_trues = run_epoch(
        model, test_loader, criterion
    )
    print(f"[nhead={nhead}] Test loss={test_loss:.4f} macro_F1={test_f1:.4f}")

    # record --------------------------------------------------------------------------
    expt = experiment_data["nhead_tuning"]["SPR_BENCH"]
    expt["metrics"]["train_macro_f1"].append(cfg_train_f1s)
    expt["metrics"]["val_macro_f1"].append(cfg_val_f1s)
    expt["metrics"]["test_macro_f1"].append(test_f1)
    expt["losses"]["train"].append(cfg_train_losses)
    expt["losses"]["val"].append(cfg_val_losses)
    expt["losses"]["test"].append(test_loss)
    expt["predictions"].append(test_preds)
    expt["ground_truth"].append(test_trues)
    expt["epochs"].append(list(range(1, num_epochs + 1)))

    # free memory before next run
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

# ------------------------------------------------------- save experiment data ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved all results to working/experiment_data.npy")
