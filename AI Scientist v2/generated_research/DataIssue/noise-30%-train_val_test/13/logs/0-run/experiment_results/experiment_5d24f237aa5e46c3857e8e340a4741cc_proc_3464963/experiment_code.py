import os, pathlib, math, time, json, random, warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- reproducibility --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- device & work dir --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- experiment data container --------------------
experiment_data = {
    "batch_size_tuning": {
        "SPR_BENCH": {
            # each batch size key will be added here
        }
    }
}


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
        d[split] = _load(f"{split}.csv")
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
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(list(s))
vocab = [PAD, UNK] + sorted(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

labels = sorted(list(set(spr["train"]["label"])))
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Num classes: {num_classes}")

# -------------------- encoding helpers --------------------
MAX_LEN = 64


def encode_seq(seq):
    ids = [stoi.get(ch, stoi[UNK]) for ch in list(seq)[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


# -------------------- PyTorch dataset --------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labs = hf_split["label"]

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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_sz, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return self.fc(x)


criterion = nn.CrossEntropyLoss()


# -------------------- training helpers --------------------
def run_epoch(model, dataloader, optimizer=None):
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
        preds.append(logits.argmax(-1).detach().cpu())
        labels.append(batch["labels"].detach().cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return (
        total_loss / len(dataloader.dataset),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


# -------------------- hyperparameter sweep --------------------
batch_sizes = [32, 64, 128, 256, 512]
EPOCHS = 6

for bs in batch_sizes:
    bs_key = str(bs)
    print(f"\n==== Training with batch_size={bs} ====")
    experiment_data["batch_size_tuning"]["SPR_BENCH"][bs_key] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    try:
        train_dl = DataLoader(
            SPRTorchDataset(spr["train"]), batch_size=bs, shuffle=True
        )
        val_dl = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=bs, shuffle=False)
        test_dl = DataLoader(SPRTorchDataset(spr["test"]), batch_size=bs, shuffle=False)

        model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optimizer)
            vl_loss, vl_f1, _, _ = run_epoch(model, val_dl, optimizer=None)

            entry = experiment_data["batch_size_tuning"]["SPR_BENCH"][bs_key]
            entry["losses"]["train"].append(tr_loss)
            entry["losses"]["val"].append(vl_loss)
            entry["metrics"]["train_f1"].append(tr_f1)
            entry["metrics"]["val_f1"].append(vl_f1)
            entry["epochs"].append(epoch)

            print(
                f"Epoch {epoch}/{EPOCHS} | bs={bs} | val_F1={vl_f1:.4f} | tr_F1={tr_f1:.4f} | {time.time()-t0:.1f}s"
            )

        # final test evaluation
        tst_loss, tst_f1, preds, gts = run_epoch(model, test_dl, optimizer=None)
        entry["predictions"] = preds.tolist()
        entry["ground_truth"] = gts.tolist()
        entry["test_f1"] = tst_f1
        print(f"Test macro-F1 with bs={bs}: {tst_f1:.4f}")

    except RuntimeError as e:
        warnings.warn(f"Skipping bs={bs} due to runtime error: {str(e)}")
        experiment_data["batch_size_tuning"]["SPR_BENCH"][bs_key]["error"] = str(e)
    finally:
        # free GPU memory
        del model, optimizer
        torch.cuda.empty_cache()

# -------------------- save experiment data --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
