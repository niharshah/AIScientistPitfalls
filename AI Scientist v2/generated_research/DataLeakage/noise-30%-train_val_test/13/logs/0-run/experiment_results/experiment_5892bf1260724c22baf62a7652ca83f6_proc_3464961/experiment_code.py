import os, pathlib, math, time, random, json
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- reproducibility --------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- experiment data container --------------------
experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------- dataset load --------------------
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
        d_split = _load(f"{split}.csv")
        d[split if split != "dev" else "dev"] = d_split
    return d


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

# -------------------- vocabulary & labels --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set(ch for seq in spr["train"]["sequence"] for ch in seq)
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
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labs = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_seq(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(encode_label(self.labs[idx]), dtype=torch.long),
        }


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(
        self, vocab, d_model=128, nhead=4, layers=2, num_classes=2, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.pos(self.embed(input_ids))
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask[..., None], 0).sum(1) / (~mask).sum(
            1, keepdim=True
        ).clamp(min=1)
        return self.fc(x)


# -------------------- helpers --------------------
def run_epoch(model, dataloader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, labels = 0.0, [], []
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
        preds.append(logits.argmax(-1).detach().cpu())
        labels.append(batch["labels"].cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (
        tot_loss / len(dataloader.dataset),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


# -------------------- hyper-parameter sweep --------------------
dropout_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
EPOCHS = 8
criterion = nn.CrossEntropyLoss()

for dr in dropout_grid:
    print(f"\n=== Training with dropout={dr} ===")
    data_key = f"dr_{dr}"
    experiment_data["dropout_rate"]["SPR_BENCH"][data_key] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, EPOCHS + 1)),
    }

    model = TransformerClassifier(vocab_size, num_classes=num_classes, dropout=dr).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        va_loss, va_f1, _, _ = run_epoch(model, val_dl, criterion)
        exp = experiment_data["dropout_rate"]["SPR_BENCH"][data_key]
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(va_loss)
        exp["metrics"]["train_f1"].append(tr_f1)
        exp["metrics"]["val_f1"].append(va_f1)
        print(
            f"Epoch {epoch} | val_loss={va_loss:.4f} val_F1={va_f1:.4f} "
            f"(train_loss={tr_loss:.4f}) [{time.time()-t0:.1f}s]"
        )
    # final test evaluation
    te_loss, te_f1, te_pred, te_gt = run_epoch(model, test_dl, criterion)
    exp["predictions"] = te_pred.tolist()
    exp["ground_truth"] = te_gt.tolist()
    exp["test_loss"] = te_loss
    exp["test_f1"] = te_f1
    print(f"Test macro-F1 (dropout={dr}): {te_f1:.4f}")

# -------------------- save --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
