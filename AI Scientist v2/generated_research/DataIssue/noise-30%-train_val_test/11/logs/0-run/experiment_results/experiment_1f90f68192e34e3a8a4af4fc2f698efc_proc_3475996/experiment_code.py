# --------------------------- No-RelVec ablation study ---------------------------
import os, math, random, time, pathlib, json, numpy as np
from typing import List, Dict

# --------------------------- experiment bookkeeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "NoRelVec": {
        "SPR_BENCH": {
            "metrics": {"train_macro_f1": [], "val_macro_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}


# --------------------------- reproducibility ------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# --------------------------- optional synthetic dataset -------------------------
import csv


def generate_synthetic_spr(root: pathlib.Path, n_train=2000, n_dev=500, n_test=700):
    root.mkdir(parents=True, exist_ok=True)
    tokens = [chr(i) for i in range(65, 91)]  # 'A'-'Z'

    def make_row(idx: int):
        length = random.randint(5, 12)
        seq_tokens = random.choices(tokens, k=length)
        seq = " ".join(seq_tokens)
        label = "evenA" if seq_tokens.count("A") % 2 == 0 else "oddA"
        return (idx, seq, label)

    def dump(split_name, n_rows):
        path = root / f"{split_name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "sequence", "label"])
            for i in range(n_rows):
                writer.writerow(make_row(i))

    dump("train", n_train)
    dump("dev", n_dev)
    dump("test", n_test)


# --------------------------- ensure dataset exists ------------------------------
DATA_PATH = pathlib.Path("./SPR_BENCH")
if not DATA_PATH.exists() or not all(
    (DATA_PATH / f).exists() for f in ["train.csv", "dev.csv", "test.csv"]
):
    print("SPR_BENCH not found – creating synthetic dataset.")
    generate_synthetic_spr(DATA_PATH)
else:
    print("SPR_BENCH found – using existing files.")

# --------------------------- device --------------------------------------------
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------- load dataset ---------------------------------------
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
    for sp, csvn in zip(["train", "dev", "test"], ["train.csv", "dev.csv", "test.csv"]):
        d[sp] = _load(csvn)
    return d


spr = load_spr_bench(DATA_PATH)
print("Split sizes:", {k: len(v) for k, v in spr.items()})

# --------------------------- vocab & encoding -----------------------------------
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
print("vocab_size:", vocab_size)

max_len = min(max(len(s.split()) for s in spr["train"]["sequence"]), 64)
print("max_len:", max_len)

label_set = sorted(list(set(spr["train"]["label"])))
label2id = {l: i for i, l in enumerate(label_set)}
num_labels = len(label2id)
print("labels:", label_set)


def encode(seq: str) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(SPRDataset(spr["train"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SPRDataset(spr["dev"]), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SPRDataset(spr["test"]), batch_size=batch_size, shuffle=False)


# --------------------------- model ----------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NoRelVecTransformer(nn.Module):
    """
    Ablation model: removes the relation-vector branch.
    Only mean-pools encoder outputs before classification.
    """

    def __init__(self, vocab_size, emb_dim, nhead, layers, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, num_labels),
        )

    def forward(self, input_ids):
        mask = input_ids == 0  # padding mask
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        # mean pool over non-pad tokens
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
        pooled = x_masked.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.classifier(pooled)


# --------------------------- helpers --------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot_loss, all_preds, all_trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if training:
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_trues


# --------------------------- training -------------------------------------------
set_seed(42)
model = NoRelVecTransformer(
    vocab_size, emb_dim=128, nhead=4, layers=2, num_labels=num_labels, dropout=0.1
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

num_epochs = 5
rec = experiment_data["NoRelVec"]["SPR_BENCH"]
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)
    rec["losses"]["train"].append(tr_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_macro_f1"].append(tr_f1)
    rec["metrics"]["val_macro_f1"].append(val_f1)
    rec["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss={val_loss:.4f} | train_F1={tr_f1:.4f} val_F1={val_f1:.4f} "
        f"(time {time.time()-t0:.1f}s)"
    )

# --------------------------- test ------------------------------------------------
test_loss, test_f1, test_preds, test_trues = run_epoch(model, test_loader, criterion)
rec["test_loss"] = test_loss
rec["test_macro_f1"] = test_f1
rec["predictions"] = test_preds
rec["ground_truth"] = test_trues
print(f"\nTest results: loss={test_loss:.4f} macro_F1={test_f1:.4f}")

# --------------------------- save experiment data -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
