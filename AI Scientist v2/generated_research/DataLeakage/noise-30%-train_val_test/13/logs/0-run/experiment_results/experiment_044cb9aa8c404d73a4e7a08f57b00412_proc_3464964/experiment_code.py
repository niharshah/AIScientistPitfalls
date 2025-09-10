import os, pathlib, math, time, json, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- misc & reproducibility --------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- experiment-data container --------------------
experiment_data = {"nhead_tuning": {"SPR_BENCH": {}}}


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
        d[split if split != "dev" else "dev"] = _load(f"{split}.csv")
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
    raise FileNotFoundError("SPR_BENCH folder not found")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocabulary & label mapping --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({ch for seq in spr["train"]["sequence"] for ch in seq})
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print("Vocab size:", vocab_size, "| Num classes:", num_classes)

# -------------------- encoding helpers --------------------
MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


# -------------------- torch dataset --------------------
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


# -------------------- model definition --------------------
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
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, dhead := nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- training helpers --------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dataloader, optimizer=None):
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


# -------------------- hyperparameter sweep --------------------
nhead_options = [2, 4, 8]
EPOCHS = 8
best_val_f1, best_nhead = -1, None

for nhead in nhead_options:
    print(f"\n==== Training with nhead={nhead} ====")
    model = TransformerClassifier(vocab_size, nhead=nhead, num_classes=num_classes).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_dict = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optimizer)
        vl_loss, vl_f1, _, _ = run_epoch(model, val_dl)
        run_dict["losses"]["train"].append(tr_loss)
        run_dict["losses"]["val"].append(vl_loss)
        run_dict["metrics"]["train_f1"].append(tr_f1)
        run_dict["metrics"]["val_f1"].append(vl_f1)
        run_dict["epochs"].append(epoch)
        print(
            f"epoch {epoch}: val_f1={vl_f1:.4f} | train_f1={tr_f1:.4f} ({time.time()-t0:.1f}s)"
        )

    # final test evaluation for this setting
    ts_loss, ts_f1, ts_preds, ts_labels = run_epoch(model, test_dl)
    run_dict["predictions"] = ts_preds.tolist()
    run_dict["ground_truth"] = ts_labels.tolist()
    run_dict["test_loss"], run_dict["test_f1"] = ts_loss, ts_f1
    print(f"Test macro-F1 with nhead={nhead}: {ts_f1:.4f}")

    experiment_data["nhead_tuning"]["SPR_BENCH"][f"nhead_{nhead}"] = run_dict

    if vl_f1 > best_val_f1:
        best_val_f1, best_nhead = vl_f1, nhead

print(f"\nBest nhead based on dev macro-F1: {best_nhead} (dev F1={best_val_f1:.4f})")

# -------------------- save --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
