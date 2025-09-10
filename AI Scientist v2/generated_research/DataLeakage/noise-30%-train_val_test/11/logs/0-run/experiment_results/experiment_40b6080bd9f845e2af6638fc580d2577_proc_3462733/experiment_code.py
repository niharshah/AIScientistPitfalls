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

import os, math, pathlib, random, time, gc, sys
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# working directory & global experiment store
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": [], "test_macro_f1": []},
        "losses": {"train": [], "val": [], "test": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# -----------------------------------------------------------------------------
# device handling  (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# dataset utilities
from datasets import load_dataset, DatasetDict


def _spr_files_ok(folder: pathlib.Path) -> bool:
    return (
        (folder / "train.csv").is_file()
        and (folder / "dev.csv").is_file()
        and (folder / "test.csv").is_file()
    )


def locate_spr_bench() -> pathlib.Path:
    """
    1) use env SPR_BENCH_DATA_DIR if set,
    2) climb upwards from cwd searching for a folder that has the 3 csv files,
    3) otherwise return None and caller will create a synthetic dataset.
    """
    env_path = os.environ.get("SPR_BENCH_DATA_DIR")
    if env_path:
        p = pathlib.Path(env_path).expanduser().resolve()
        if _spr_files_ok(p):
            return p
    # search upwards for at most 5 parent levels
    here = pathlib.Path.cwd()
    for _ in range(6):
        if _spr_files_ok(here / "SPR_BENCH"):
            return (here / "SPR_BENCH").resolve()
        here = here.parent
    return None


def build_dummy_csv(path: pathlib.Path, n_rows: int):
    import csv

    random.seed(42)
    tokens = ["A", "B", "C", "D"]
    labels = ["X", "Y"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence", "label"])
        for i in range(n_rows):
            seq = " ".join(random.choices(tokens, k=random.randint(3, 10)))
            lab = random.choice(labels)
            writer.writerow([i, seq, lab])


def ensure_dataset_available() -> pathlib.Path:
    loc = locate_spr_bench()
    if loc is not None:
        print(f"Found SPR_BENCH at {loc}")
        return loc

    # fallback – make tiny synthetic dataset
    print("WARNING: SPR_BENCH not found. Creating a tiny synthetic dataset.")
    synth_dir = pathlib.Path(working_dir) / "SPR_BENCH_SYNTH"
    synth_dir.mkdir(exist_ok=True)
    build_dummy_csv(synth_dir / "train.csv", 400)
    build_dummy_csv(synth_dir / "dev.csv", 100)
    build_dummy_csv(synth_dir / "test.csv", 200)
    return synth_dir


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


data_path = ensure_dataset_available()
spr = load_spr_bench(data_path)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# -----------------------------------------------------------------------------
# simple whitespace tokenizer
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
print("Vocab size:", vocab_size)


def encode(seq: str, max_len: int) -> List[int]:
    tok_ids = [vocab.get(t, vocab[UNK]) for t in seq.strip().split()][:max_len]
    tok_ids += [vocab[PAD]] * (max_len - len(tok_ids))
    return tok_ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
print("Maximum sequence length:", max_len)

label_set = sorted(set(spr["train"]["label"]))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)
print("Number of labels:", num_labels)


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode(self.seqs[idx], max_len), dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# -----------------------------------------------------------------------------
# model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.pos_enc(self.embedding(input_ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# training helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, train_flag, optimizer=None):
    model.train() if train_flag else model.eval()
    total_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        if train_flag:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_flag):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_flag:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return total_loss / len(loader.dataset), macro_f1, all_preds, all_trues


# -----------------------------------------------------------------------------
# hyper-parameter tuning (number of epochs)
epoch_grid = [5, 10, 15]  # slightly smaller grid for faster turnaround

for n_epochs in epoch_grid:
    print(f"\n========== Training for {n_epochs} epochs ==========")
    model = CharTransformer(vocab_size, 128, 8, 2, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_f1, _, _ = run_epoch(
            model, train_loader, True, optimizer=optimizer
        )
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, False)

        # logging
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
        experiment_data["SPR_BENCH"]["epochs"].append(epoch)

        print(
            f"Epoch {epoch}/{n_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={train_f1:.4f} val_F1={val_f1:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # final test run for this training duration
    test_loss, test_f1, preds, trues = run_epoch(model, test_loader, False)
    print(f"Test after {n_epochs} epochs: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

    experiment_data["SPR_BENCH"]["losses"]["test"].append(test_loss)
    experiment_data["SPR_BENCH"]["metrics"]["test_macro_f1"].append(test_f1)
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(trues)

    # cleanup to avoid GPU OOM between runs
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll experiment data saved to {working_dir}/experiment_data.npy")
