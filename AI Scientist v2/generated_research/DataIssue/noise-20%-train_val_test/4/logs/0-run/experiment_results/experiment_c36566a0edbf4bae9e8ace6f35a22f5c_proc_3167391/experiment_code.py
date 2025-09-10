import os, pathlib, time, numpy as np, torch
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ----------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- experiment store -----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": []},
        "losses": {"train": [], "val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- dataset loader -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # change if needed
spr = load_spr_bench(DATA_PATH)


# ----------------- vocab build -----------------
def build_vocab(ds):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in ds["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id, unk_id = vocab["<PAD>"], vocab["<UNK>"]
vocab_size = len(vocab)
num_labels = len(set(spr["train"]["label"]))
print(f"Vocab size = {vocab_size} | #labels = {num_labels}")


def encode(seq):
    return [vocab.get(c, unk_id) for c in seq]


for split in ("train", "dev", "test"):
    spr[split] = spr[split].map(lambda x: {"input_ids": encode(x["sequence"])})


# ----------------- collate fn -----------------
def collate_fn(batch):
    ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = seq
        attn[i, : len(seq)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )
    for split in ("train", "dev", "test")
}


# ----------------- model -----------------
class CountAwareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=6,
        dim_feedforward=512,
        dropout=0.2,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(4096, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        # count pathway
        self.count_proj = nn.Sequential(
            nn.Linear(vocab_size, d_model // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_model + d_model // 2, num_labels)

    def forward(self, input_ids, attention_mask, counts):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        # mean pool on valid tokens
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        count_feat = self.count_proj(counts)
        concat = torch.cat([pooled, count_feat], dim=-1)
        return self.classifier(concat)


# helper to compute count vectors inside batch (one-hot counts, float normalised)
def batch_counts(input_ids):
    # input_ids: (B, L)
    B, L = input_ids.size()
    flat = input_ids.view(-1)
    one_hot = torch.zeros(B * L, vocab_size, device=input_ids.device)
    valid = flat != pad_id
    one_hot[torch.arange(B * L, device=input_ids.device), flat] = 1.0
    one_hot = one_hot[valid]
    idx = torch.arange(B, device=input_ids.device).repeat_interleave(
        (valid.view(B, L)).sum(1)
    )
    counts = torch.zeros(B, vocab_size, device=input_ids.device).index_add_(
        0, idx, one_hot
    )
    counts = counts / counts.sum(1, keepdim=True).clamp_min(1.0)  # normalise
    return counts


# ----------------- training utils -----------------
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        counts = batch_counts(batch["input_ids"])
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["attention_mask"], counts)
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ----------------- training loop -----------------
model = CountAwareTransformer(vocab_size, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15 * len(loaders["train"])
)

best_val_f1, best_state = 0.0, os.path.join(working_dir, "best_model.pt")
epochs = 15
for epoch in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(
        model, loaders["train"], criterion, optimizer, scheduler
    )
    val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"], criterion)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_F1={val_f1:.4f}"
    )
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_state)

# ----------------- final test evaluation -----------------
model.load_state_dict(torch.load(best_state))
test_loss, test_f1, test_preds, test_gts = run_epoch(model, loaders["test"], criterion)
experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["metrics"]["test_f1"] = test_f1
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
print(f"Best Dev Macro-F1: {best_val_f1:.4f} | Test Macro-F1: {test_f1:.4f}")

# ----------------- save experiment -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
