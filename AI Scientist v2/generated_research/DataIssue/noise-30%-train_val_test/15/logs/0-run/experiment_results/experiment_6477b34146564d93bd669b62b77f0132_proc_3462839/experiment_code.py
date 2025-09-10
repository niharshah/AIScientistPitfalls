import os, pathlib, random, math, time, json, numpy as np, torch
from typing import List, Dict
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# -------------- house-keeping & reproducibility --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
print(f"Using device: {device}")


# -------------- load SPR_BENCH or synthetic fallback ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    dd["train"] = _load("train.csv")
    dd["dev"] = _load("dev.csv")
    dd["test"] = _load("test.csv")
    return dd


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:
    print("SPR_BENCH not found, generating synthetic data …")

    def synth_split(n_rows, n_labels=5, max_len=20):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDEXYZUVW")
        for i in range(n_rows):
            seq = "".join(random.choices(alphabet, k=random.randint(5, max_len)))
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(random.randint(0, n_labels - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict(
        {"train": synth_split(500), "dev": synth_split(100), "test": synth_split(100)}
    )

num_labels = len(set(spr["train"]["label"]))
print(f"Loaded dataset with {num_labels} labels.")

# -------------- vocab & encoding ----------------------------------------
PAD_ID = 0


def build_vocab(dataset) -> Dict[str, int]:
    chars = set("".join(dataset["sequence"]))
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
id2char = {i: c for c, i in vocab.items()}
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# -------------- dataset wrappers ----------------------------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seqs[idx], self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_ds = SPRTorchDataset(spr["train"], MAX_LEN)
dev_ds = SPRTorchDataset(spr["dev"], MAX_LEN)
test_ds = SPRTorchDataset(spr["test"], MAX_LEN)


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


# -------------- model definition ----------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SPRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=(attention_mask == 0))
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.cls(x)


# -------------- hyper-parameter grid ------------------------------------
LR_CANDIDATES = [3e-4, 5e-4, 7e-4, 1e-3, 2e-3]
EPOCHS = 5
BATCH_TRAIN = 128
BATCH_EVAL = 256

# -------------- experiment data dict ------------------------------------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {"runs": []}  # each element will be a dict of a single LR run
    }
}

# -------------- helpers -------------------------------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, train_mode, optimizer=None):
    model.train() if train_mode else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# data loaders (re-create with right batch_size once)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_TRAIN, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate
)

# -------------- grid search loop ----------------------------------------
for lr in LR_CANDIDATES:
    print(f"\n########## Training with learning_rate={lr:.4g} ##########")
    model = SPRTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_record = {
        "lr": lr,
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "test_loss": None,
        "test_macroF1": None,
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, True, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_loader, False)

        run_record["losses"]["train"].append(tr_loss)
        run_record["losses"]["val"].append(val_loss)
        run_record["metrics"]["train"].append(tr_f1)
        run_record["metrics"]["val"].append(val_f1)
        run_record["epochs"].append(epoch)

        print(
            f"Epoch {epoch}: lr={lr:.4g}  train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}"
        )

    # final test evaluation
    test_loss, test_f1, test_preds, test_gts = run_epoch(model, test_loader, False)
    run_record["test_loss"] = test_loss
    run_record["test_macroF1"] = test_f1
    run_record["predictions"] = test_preds
    run_record["ground_truth"] = test_gts
    print(f"Test : loss={test_loss:.4f} MacroF1={test_f1:.4f}")

    experiment_data["learning_rate"]["SPR_BENCH"]["runs"].append(run_record)

    # free GPU memory
    del model
    torch.cuda.empty_cache()

# -------------- save results --------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
