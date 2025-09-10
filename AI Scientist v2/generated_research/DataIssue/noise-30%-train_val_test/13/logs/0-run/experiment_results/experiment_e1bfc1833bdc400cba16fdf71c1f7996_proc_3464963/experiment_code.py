import os, pathlib, math, time, json, random, gc
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# -------------------- reproducibility --------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- device & working dir --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- experiment data container --------------------
experiment_data = {"weight_decay": {}}


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
    raise FileNotFoundError("SPR_BENCH folder not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocabulary & labels --------------------
PAD, UNK = "<PAD>", "<UNK>"
vocab = set()
[vocab.update(list(s)) for s in spr["train"]["sequence"]]
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
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def encode_label(lab):
    return label2id[lab]


# -------------------- dataset --------------------
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
train_dl_full = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_dl_full = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=batch_size, shuffle=False
)
test_dl_full = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=batch_size, shuffle=False
)


# -------------------- model --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 256, 0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.pos(self.embed(input_ids))
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)


# -------------------- train/val helpers --------------------
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
        labels.append(batch["labels"].detach().cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (
        tot_loss / len(dataloader.dataset),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


# -------------------- hyperparameter search --------------------
search_wd = [0.0, 1e-5, 1e-4, 1e-3]
EPOCHS = 8
for wd in search_wd:
    key = f"wd_{wd}"
    print(f"\n================ Weight Decay = {wd} =================")
    experiment_data["weight_decay"][key] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    model = TransformerClassifier(vocab_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    best_val_f1 = -1
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl_full, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_dl_full, criterion)
        exp = experiment_data["weight_decay"][key]
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train_f1"].append(tr_f1)
        exp["metrics"]["val_f1"].append(val_f1)
        exp["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f} val_F1={val_f1:.4f} "
            f"(train_loss={tr_loss:.4f}) [{time.time()-t0:.1f}s]"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    # final test evaluation
    test_loss, test_f1, test_preds, test_labels = run_epoch(
        model, test_dl_full, criterion
    )
    exp["predictions"] = test_preds.tolist()
    exp["ground_truth"] = test_labels.tolist()
    exp["test_f1"] = test_f1
    exp["test_loss"] = test_loss
    print(f"Test macro-F1 (wd={wd}): {test_f1:.4f}")
    # cleanup
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

# -------------------- save experiment data --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
