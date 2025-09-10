import os, pathlib, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset

# ---------- saving dict ----------
experiment_data = {
    "dropout_sweep": {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "dropouts": [],
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
num_labels = len(set(spr["train"]["label"]))
print(f"Loaded SPR_BENCH with {len(spr['train'])} train examples.")


# ---------- vocab ----------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id = vocab["<PAD>"]
unk_id = vocab["<UNK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------- encode sequences ----------
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------- collate ----------
def collate_fn(batch):
    ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : seq.size(0)] = seq
        attn[i, : seq.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        nlayers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.in_drop = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.in_drop(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


criterion = nn.CrossEntropyLoss()


# ---------- training helpers ----------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- sweep ----------
dropout_values = [0.0, 0.1, 0.15, 0.25, 0.35]
epochs = 15

for p_drop in dropout_values:
    print(f"\n=== Training with dropout={p_drop} ===")
    model = CharTransformer(vocab_size, num_labels, dropout=p_drop).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_f1, best_state = 0.0, None
    tr_losses, val_losses, tr_f1s, val_f1s = [], [], [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, loaders["train"], optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"])
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        tr_f1s.append(tr_f1)
        val_f1s.append(val_f1)
        print(
            f"  Epoch {epoch:02d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    # reload best and test
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, loaders["test"])

    # save run data
    ed = experiment_data["dropout_sweep"]["SPR_BENCH"]
    ed["dropouts"].append(p_drop)
    ed["losses"]["train"].append(tr_losses)
    ed["losses"]["val"].append(val_losses)
    ed["metrics"]["train_f1"].append(tr_f1s)
    ed["metrics"]["val_f1"].append(val_f1s)
    ed["predictions"].append(preds)
    ed["ground_truth"].append(gts)
    ed["epochs"].append(list(range(1, epochs + 1)))

    print(f"Best val_F1={best_f1:.4f} | Test_F1={test_f1:.4f}")

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
