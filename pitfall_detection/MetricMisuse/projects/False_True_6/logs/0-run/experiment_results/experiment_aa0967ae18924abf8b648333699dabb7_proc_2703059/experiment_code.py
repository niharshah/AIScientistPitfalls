import os, pathlib, time, math, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------------------------------- house-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------- experiment log
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
exp_rec = experiment_data["SPR_BENCH"]


# --------------------------------------------------- metric helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / max(sum(w), 1)


# --------------------------------------------------- data loading
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})

# --------------------------------------------------- vocab
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    for tok in sorted(
        set(itertools.chain.from_iterable(seq.strip().split() for seq in dataset))
    ):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dsets["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


label_set = sorted(set(dsets["train"]["label"]))
lab2idx = {l: i for i, l in enumerate(label_set)}
idx2lab = {i: l for l, i in lab2idx.items()}
print("Labels:", label_set)


# --------------------------------------------------- torch Dataset
class SPRTorchDS(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = [lab2idx[l] for l in split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        return {
            "ids": torch.tensor(encode(s), dtype=torch.long),
            "lab": torch.tensor(self.lab[idx], dtype=torch.long),
            "shape_cnt": torch.tensor(count_shape_variety(s), dtype=torch.float),
            "raw": s,
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    labs = torch.stack([b["lab"] for b in batch])
    shapes = torch.stack([b["shape_cnt"] for b in batch]).unsqueeze(-1)  # (B,1)
    pad_ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=vocab[PAD])
    return {
        "input_ids": pad_ids,
        "labels": labs,
        "shape_cnt": shapes,
        "raw_seq": [b["raw"] for b in batch],
    }


train_loader = DataLoader(
    SPRTorchDS(dsets["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDS(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorchDS(dsets["test"]), batch_size=256, shuffle=False, collate_fn=collate
)


# --------------------------------------------------- model
class SymbolicTransformer(nn.Module):
    def __init__(
        self, vocab_size, n_labels, d_model=64, nhead=4, num_layers=2, dim_ff=128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(256, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.sym_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Linear(d_model + 16, n_labels)

    def forward(self, ids, shape_cnt):
        B, L = ids.size()
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.embed(ids) + self.pos_embed(pos)
        mask = ids.eq(0)  # PAD mask
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).sum(1) / (~mask).sum(1).clamp(
            min=1
        ).unsqueeze(
            -1
        )  # mean pool
        sym = self.sym_mlp(shape_cnt)
        logits = self.out(torch.cat([x, sym], dim=-1))
        return logits


model = SymbolicTransformer(len(vocab), len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, all_seq, all_t, all_p = 0.0, [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        shp = batch["shape_cnt"].to(device)
        lab = batch["labels"].to(device)
        logits = model(ids, shp)
        loss = criterion(logits, lab)
        total_loss += loss.item() * len(lab)
        pred = logits.argmax(-1)
        all_seq.extend(batch["raw_seq"])
        all_t.extend(lab.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    swa = shape_weighted_accuracy(all_seq, all_t, all_p)
    return total_loss / len(all_t), swa, all_p, all_t, all_seq


# --------------------------------------------------- training loop
MAX_EPOCHS, PATIENCE = 25, 4
best_val_swa, best_state, epochs_no_imp = -1.0, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        shp = batch["shape_cnt"].to(device)
        lab = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids, shp)
        loss = criterion(logits, lab)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(lab)
    train_loss = running / len(train_loader.dataset)
    val_loss, val_swa, *_ = evaluate(dev_loader)

    print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  SWA={val_swa:.3f}")

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train"].append({"epoch": epoch})
    exp_rec["metrics"]["val"].append({"epoch": epoch, "swa": val_swa})
    exp_rec["timestamps"].append(time.time())

    if val_swa > best_val_swa + 1e-4:
        best_val_swa, best_state = val_swa, model.state_dict()
        epochs_no_imp = 0
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------------------------------- final test
test_loss, test_swa, preds, trues, seqs = evaluate(test_loader)
print(f"\nTEST  SWA={test_swa:.3f}")

exp_rec["predictions"], exp_rec["ground_truth"] = preds, trues
exp_rec["metrics"]["test"] = {"swa": test_swa, "loss": test_loss}

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
