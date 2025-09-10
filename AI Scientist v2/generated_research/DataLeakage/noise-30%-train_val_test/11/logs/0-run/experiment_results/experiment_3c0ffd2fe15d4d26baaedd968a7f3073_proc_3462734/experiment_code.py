import os, math, pathlib, random, time, gc
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# -------------------------- bookkeeping ---------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "batch_size_tuning": {"SPR_BENCH": {}}  # will be filled with results per batch size
}

# --------------------------- reproducibility -----------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- dataset -------------------------------------------------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split if split != "dev" else "dev"] = _load(f"{split}.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

# ---------------------------- vocab / encoding ---------------------------------------
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
    toks = seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
print("Max seq len:", max_len)

label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
num_labels = len(label2id)
print("Num labels:", num_labels)


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


# --------------------------- model ----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, layers, num_labels, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            emb_dim, nhead, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Linear(emb_dim, num_labels)

    def forward(self, ids):
        mask = ids == 0
        x = self.emb(ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.cls(x)


# --------------------------- training utils ------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_flag = optimizer is not None
    model.train() if train_flag else model.eval()
    tot_loss, preds_all, trues_all = 0.0, [], []
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
        preds_all.extend(logits.argmax(-1).detach().cpu().numpy())
        trues_all.extend(batch["labels"].detach().cpu().numpy())
    avg_loss = tot_loss / len(loader.dataset)
    f1 = f1_score(trues_all, preds_all, average="macro")
    return avg_loss, f1, preds_all, trues_all


# --------------------------- hyperparameter sweep ------------------------------------
batch_sizes = [32, 64, 128]
num_epochs = 10

for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    # loaders
    train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=bs, shuffle=True)
    val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=bs, shuffle=False)
    test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=bs, shuffle=False)

    # model / optimiser
    model = CharTransformer(vocab_size, 128, 8, 2, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # record containers
    res = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)

        res["losses"]["train"].append(tr_loss)
        res["losses"]["val"].append(val_loss)
        res["metrics"]["train_macro_f1"].append(tr_f1)
        res["metrics"]["val_macro_f1"].append(val_f1)
        res["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    # final test evaluation
    test_loss, test_f1, preds, trues = run_epoch(model, test_loader, criterion)
    res["losses"]["test"] = test_loss
    res["metrics"]["test_macro_f1"] = test_f1
    res["predictions"] = preds
    res["ground_truth"] = trues
    print(f"Test: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

    # store
    experiment_data["batch_size_tuning"]["SPR_BENCH"][str(bs)] = res

    # clean up GPU memory before next sweep value
    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

# --------------------------- save results --------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "\nSaved all experiment data to", os.path.join(working_dir, "experiment_data.npy")
)
