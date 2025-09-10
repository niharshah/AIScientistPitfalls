import os, pathlib, math, time, json, random, warnings

warnings.filterwarnings("ignore")

# -------------------- reproducibility --------------------
SEED = 42
random.seed(SEED)
import numpy as np, torch

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------- experiment container --------------------
experiment_data = {"batch_size": {}}  # <-- top-level key for this tuning type


# -------------------- dataset helpers --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab / labels --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(list(s))
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)

MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    return ids + [stoi[PAD]] * (MAX_LEN - len(ids))


def encode_label(lab):
    return label2id[lab]


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labs = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(encode_label(self.labs[idx]), dtype=torch.long),
        }


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
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        mask = ids == 0
        x = self.pos(self.embed(ids))
        x = self.tr(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- training / evaluation --------------------
def run_epoch(model, dataloader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, labs = 0.0, [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labs.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labs = torch.cat(labs).numpy()
    return (
        tot_loss / len(dataloader.dataset),
        f1_score(labs, preds, average="macro"),
        preds,
        labs,
    )


# -------------------- hyperparameter search --------------------
batch_sizes = [64, 128, 256, 512]
EPOCHS = 8

for bs in batch_sizes:
    print(f"\n===== Training with batch size {bs} =====")
    train_dl = DataLoader(SPRTorchDataset(spr["train"]), batch_size=bs, shuffle=True)
    val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=bs, shuffle=False)
    test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=bs, shuffle=False)

    model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_f1, _, _ = run_epoch(model, val_dl, criterion)
        record["losses"]["train"].append(tr_loss)
        record["losses"]["val"].append(vl_loss)
        record["metrics"]["train"].append(tr_f1)
        record["metrics"]["val"].append(vl_f1)
        record["epochs"].append(epoch)
        print(
            f"Epoch {epoch}/{EPOCHS}  bs={bs}  val_F1={vl_f1:.4f}  "
            f"train_loss={tr_loss:.3f}  time={time.time()-t0:.1f}s"
        )

    # final test evaluation
    tst_loss, tst_f1, preds, gts = run_epoch(model, test_dl, criterion)
    record["predictions"], record["ground_truth"] = preds.tolist(), gts.tolist()
    record["test_f1"], record["test_loss"] = tst_f1, tst_loss

    # store
    experiment_data["batch_size"].setdefault("SPR_BENCH", {})[f"bs_{bs}"] = record
    print(f"Batch size {bs}: Test macro-F1 = {tst_f1:.4f}")

# -------------------- save experiment --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nAll experiments finished and saved.")
