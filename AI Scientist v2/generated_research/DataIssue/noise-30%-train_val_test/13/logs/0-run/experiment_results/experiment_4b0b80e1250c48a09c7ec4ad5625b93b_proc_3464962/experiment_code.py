import os, pathlib, math, time, random, json
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- directories / device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment container --------------------
experiment_data = {"num_layers": {"SPR_BENCH": {}}}  # hyper-parameter tuned


# -------------------- load SPR_BENCH --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):  # tiny helper
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


data_root_candidates = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
for p in data_root_candidates:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab / labels --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({ch for s in spr["train"]["sequence"] for ch in s})
stoi = {ch: i for i, ch in enumerate(vocab)}
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
itos = {i: ch for ch, i in stoi.items()}
vocab_size, num_classes = len(vocab), len(label2id)
MAX_LEN = 64
print(f"Vocab size: {vocab_size}  Num classes: {num_classes}")


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labs = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=batch_size)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids.eq(0)
        x = self.pos(self.embed(input_ids))
        x = self.transformer(x, src_key_padding_mask=mask)
        x.masked_fill_(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- train / eval helpers --------------------
def run_epoch(model, dataloader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (
        total_loss / len(dataloader.dataset),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


# -------------------- hyper-parameter sweep --------------------
EPOCHS = 8
for nl in [1, 2, 3, 4]:
    print(f"\n=== Training with num_layers={nl} ===")
    model = TransformerClassifier(vocab_size, 128, 4, nl, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data_slot = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_f1, _, _ = run_epoch(model, val_dl, criterion)
        data_slot["losses"]["train"].append(tr_loss)
        data_slot["losses"]["val"].append(vl_loss)
        data_slot["metrics"]["train_f1"].append(tr_f1)
        data_slot["metrics"]["val_f1"].append(vl_f1)
        data_slot["epochs"].append(epoch)
        print(
            f"  Ep{epoch}: val_loss={vl_loss:.4f} val_F1={vl_f1:.4f} (train_F1={tr_f1:.4f}) [{time.time()-t0:.1f}s]"
        )

    # final test
    ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(model, test_dl, criterion)
    data_slot["predictions"], data_slot["ground_truth"] = (
        ts_preds.tolist(),
        ts_labels.tolist(),
    )
    print(f"  --> Test macro-F1={ts_f1:.4f}")

    experiment_data["num_layers"]["SPR_BENCH"][f"nl_{nl}"] = data_slot
    # free memory before next run
    del model
    torch.cuda.empty_cache()

# -------------------- save --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
