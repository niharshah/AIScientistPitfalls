# num_epochs hyper-parameter tuning on SPR_BENCH
import os, pathlib, math, time, random, json
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- reproducibility --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

# -------------------- experiment dict --------------------
experiment_data = {
    "num_epochs": {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):  # csv loader helper
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    ds = DatasetDict()
    for split_file, split_name in [
        ("train.csv", "train"),
        ("dev.csv", "dev"),
        ("test.csv", "test"),
    ]:
        ds[split_name] = _load(split_file)
    return ds


data_root = None
for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        data_root = p
        break
if data_root is None:
    raise FileNotFoundError("SPR_BENCH folder not found")

spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = {ch for seq in spr["train"]["sequence"] for ch in seq}
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)

# -------------------- labels --------------------
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)

# -------------------- encoders --------------------
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
    def __init__(self, vocab, d_model=128, nhead=4, layers=2, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, ids):
        mask = ids == 0
        x = self.pos(self.embed(ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


model = TransformerClassifier(vocab_size, n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------- train / eval --------------------
def run_epoch(dloader, train_mode=True):
    model.train() if train_mode else model.eval()
    tot_loss, preds, labs = 0.0, [], []
    for batch in dloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train_mode):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.append(logits.argmax(-1).cpu())
        labs.append(batch["labels"].cpu())
    preds, labs = torch.cat(preds).numpy(), torch.cat(labs).numpy()
    macro_f1 = f1_score(labs, preds, average="macro")
    return tot_loss / len(dloader.dataset), macro_f1, preds, labs


# -------------------- training loop with early stopping --------------------
MAX_EPOCHS, PATIENCE = 30, 5
best_f1, patience_cnt = -1.0, 0
best_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_dl, True)
    val_loss, val_f1, _, _ = run_epoch(val_dl, False)
    ed = experiment_data["num_epochs"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f}  val_F1={val_f1:.4f} "
        f"(train_loss={tr_loss:.4f})  [{time.time()-t0:.1f}s]"
    )
    # early stopping
    if val_f1 > best_f1:
        best_f1, patience_cnt, best_state = val_f1, 0, model.state_dict()
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------- test evaluation (best model) --------------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_labels = run_epoch(test_dl, False)
ed = experiment_data["num_epochs"]["SPR_BENCH"]
ed["predictions"] = test_preds.tolist()
ed["ground_truth"] = test_labels.tolist()
print(f"Best validation F1: {best_f1:.4f} | Test F1: {test_f1:.4f}")

# -------------------- save --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
