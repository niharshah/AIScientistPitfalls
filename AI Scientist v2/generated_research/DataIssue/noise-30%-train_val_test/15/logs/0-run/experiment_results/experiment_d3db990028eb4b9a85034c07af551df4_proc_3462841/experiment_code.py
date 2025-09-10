import os, pathlib, random, time, math, json, numpy as np, torch
from typing import List, Dict
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ----------------------- experiment data dict ---------------------------
experiment_data = {
    "weight_decay_tuning": {
        "SPR_BENCH": {
            "wd_values": [],
            "metrics": {"train": [], "val": []},  # list of lists per wd
            "losses": {"train": [], "val": []},
            "predictions": [],  # list per wd
            "ground_truth": [],  # list per wd
            "test_loss": [],
            "test_macroF1": [],
            "epochs": [],
        }
    }
}

# ----------------------- house-keeping & GPU -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- load SPR_BENCH or fallback ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


dataset_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if dataset_path.exists():
    spr = load_spr_bench(dataset_path)
else:
    print("SPR_BENCH not found, generating synthetic data …")

    def synth_split(n_rows, n_labels=5, max_len=20):
        data = {"id": [], "sequence": [], "label": []}
        alphabet = list("ABCDEXYZUVW")
        for i in range(n_rows):
            length = random.randint(5, max_len)
            seq = "".join(random.choices(alphabet, k=length))
            label = random.randint(0, n_labels - 1)
            data["id"].append(str(i))
            data["sequence"].append(seq)
            data["label"].append(label)
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict(
        {"train": synth_split(500), "dev": synth_split(100), "test": synth_split(100)}
    )

num_labels = len(set(spr["train"]["label"]))
print(f"Loaded dataset with {num_labels} labels.")

# ----------------------- vocabulary & encoding ---------------------------
PAD_ID = 0


def build_vocab(dataset) -> Dict[str, int]:
    chars = set()
    for s in dataset["sequence"]:
        chars.update(list(s))
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # reserve 0 for PAD
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# ----------------------- dataset wrapper ---------------------------------
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
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": label}


train_ds = SPRTorchDataset(spr["train"], MAX_LEN)
dev_ds = SPRTorchDataset(spr["dev"], MAX_LEN)
test_ds = SPRTorchDataset(spr["test"], MAX_LEN)


def collate(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ----------------------- model ------------------------------------------
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ----------------------- training helpers --------------------------------
criterion = nn.CrossEntropyLoss()
EPOCHS = 5


def run_epoch(model, loader, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ----------------------- weight-decay sweep ------------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]

for wd in weight_decays:
    print(f"\n==== Training with weight_decay={wd} ====")
    experiment_data["weight_decay_tuning"]["SPR_BENCH"]["wd_values"].append(wd)

    model = SPRTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)

    tr_losses, val_losses, tr_f1s, val_f1s = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, True, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_loader, False)

        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_f1s.append(tr_f1)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}"
        )

    # store epoch data
    dstore = experiment_data["weight_decay_tuning"]["SPR_BENCH"]
    dstore["losses"]["train"].append(tr_losses)
    dstore["losses"]["val"].append(val_losses)
    dstore["metrics"]["train"].append(tr_f1s)
    dstore["metrics"]["val"].append(val_f1s)
    dstore["epochs"].append(list(range(1, EPOCHS + 1)))

    # final test evaluation
    test_loss, test_f1, test_preds, test_gts = run_epoch(model, test_loader, False)
    print(f"Test : loss={test_loss:.4f} MacroF1={test_f1:.4f}")

    dstore["test_loss"].append(test_loss)
    dstore["test_macroF1"].append(test_f1)
    dstore["predictions"].append(test_preds)
    dstore["ground_truth"].append(test_gts)

# ----------------------- save experiment data ---------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
