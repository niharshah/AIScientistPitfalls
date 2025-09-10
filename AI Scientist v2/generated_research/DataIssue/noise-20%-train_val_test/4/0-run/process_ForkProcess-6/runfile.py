import os, pathlib, json, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict

# ---------- experiment data ----------
experiment_data = {
    "num_epochs": {  # <-- hyper-parameter we sweep
        "SPR_BENCH": {}  # every epoch budget gets its own sub-dict below
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- dataset loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(file):  # small helper
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# you can change SPR_PATH with an env variable if needed
data_root = pathlib.Path(os.environ.get("SPR_PATH", "./SPR_BENCH"))
spr = load_spr_bench(data_root)
num_labels = len(set(spr["train"]["label"]))
print("Loaded SPR_BENCH with", len(spr["train"]), "train examples.")


# ---------- build vocab ----------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id, unk_id, vocab_size = vocab["<PAD>"], vocab["<UNK>"], len(vocab)
print("Vocab size:", vocab_size)


# ---------- encode sequences ----------
def encode(seq):  # char to id
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
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


criterion = nn.CrossEntropyLoss()


# ---------- helpers ----------
def run_epoch(model, loader, is_train, optimizer=None):
    model.train() if is_train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
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


# ---------- hyper-parameter sweep ----------
epoch_candidates = [10, 20, 30]

for epochs in epoch_candidates:
    print(f"\n=== Training with {epochs} epochs ===")
    run_key = str(epochs)  # use as dict key
    experiment_data["num_epochs"]["SPR_BENCH"][run_key] = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = CharTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_f1, best_path = 0.0, os.path.join(working_dir, f"best_{epochs}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(
            model, loaders["train"], is_train=True, optimizer=optimizer
        )
        val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"], is_train=False)
        exp = experiment_data["num_epochs"]["SPR_BENCH"][run_key]
        exp["epochs"].append(epoch)
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train_f1"].append(tr_f1)
        exp["metrics"]["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_F1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    # ---------- test on best checkpoint ----------
    model.load_state_dict(torch.load(best_path))
    test_loss, test_f1, test_preds, test_gts = run_epoch(
        model, loaders["test"], is_train=False
    )

    print(
        f"Finished {epochs} epochs: Best Val F1={best_f1:.4f} | "
        f"Test F1={test_f1:.4f}"
    )

    exp = experiment_data["num_epochs"]["SPR_BENCH"][run_key]
    exp["test_f1"] = test_f1
    exp["predictions"] = test_preds
    exp["ground_truth"] = test_gts

# ---------- save everything ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
