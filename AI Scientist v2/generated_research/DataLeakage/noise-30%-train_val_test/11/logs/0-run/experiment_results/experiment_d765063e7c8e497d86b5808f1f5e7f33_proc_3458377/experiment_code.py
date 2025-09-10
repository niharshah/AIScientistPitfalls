import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ 1. Dataset loading ============ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})


# ============ 2. Tokenisation & vocab ============ #
def simple_tokenize(seq: str):
    seq = seq.strip()
    return seq.split() if " " in seq else list(seq)


# build vocab
all_tokens = set()
for s in dsets["train"]["sequence"]:
    all_tokens.update(simple_tokenize(s))
specials = ["<pad>", "<unk>"]
idx2tok = specials + sorted(all_tokens)
tok2idx = {t: i for i, t in enumerate(idx2tok)}
pad_idx = tok2idx["<pad>"]
unk_idx = tok2idx["<unk>"]

# label mapping
labels_set = sorted(set(dsets["train"]["label"]))
lbl2idx = {l: i for i, l in enumerate(labels_set)}
num_classes = len(labels_set)
print(f"Vocab size: {len(tok2idx)}, classes: {num_classes}")


def encode_sequence(seq):
    return [tok2idx.get(t, unk_idx) for t in simple_tokenize(seq)]


class SPRTorchDataset(Dataset):
    def __init__(self, hf_subset):
        self.seqs = hf_subset["sequence"]
        self.labels = [lbl2idx[l] for l in hf_subset["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    max_len = max(len(i) for i in ids)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    attn_mask = torch.zeros_like(padded)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = seq
        attn_mask[i, : len(seq)] = 1
    return {"input_ids": padded, "attention_mask": attn_mask, "labels": labels}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(dsets["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    SPRTorchDataset(dsets["dev"]),
    batch_size=batch_size * 2,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRTorchDataset(dsets["test"]),
    batch_size=batch_size * 2,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============ 3. Model definition ============ #
class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, nlayers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)  # B,L,E
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        # simple mean pooling over valid tokens
        summed = (x * attention_mask.unsqueeze(-1)).sum(1)
        lengths = attention_mask.sum(1, keepdim=True)
        pooled = summed / lengths
        return self.classifier(pooled)


model = TinyTransformer(len(tok2idx), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


# ============ 4. Training loop ============ #
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    losses, ys_true, ys_pred = [], [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        ys_true.append(batch["labels"].cpu().numpy())
        ys_pred.append(logits.argmax(-1).detach().cpu().numpy())
    y_true = np.concatenate(ys_true)
    y_pred = np.concatenate(ys_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return np.mean(losses), macro_f1, y_true, y_pred


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)
    dt = time.time() - t0
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | "
        f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}  [{dt:.1f}s]"
    )
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)

# ============ 5. Final test evaluation ============ #
test_loss, test_f1, y_true, y_pred = run_epoch(test_loader, train=False)
print(f"TEST: loss={test_loss:.4f} macro_f1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = y_pred
experiment_data["SPR_BENCH"]["ground_truth"] = y_true
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
