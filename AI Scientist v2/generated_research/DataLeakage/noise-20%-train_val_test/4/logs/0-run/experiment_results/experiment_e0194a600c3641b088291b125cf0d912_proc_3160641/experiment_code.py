import os, pathlib, math, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict


# ------------- reproducibility -------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ------------- experiment log -------------
experiment_data = {
    "nhead_tuning": {
        "SPR_BENCH": {
            "results": {},  # will hold dict per nhead value
        }
    }
}

# ------------- working dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------- dataset -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
num_labels = len(set(spr["train"]["label"]))
print("Loaded SPR_BENCH with", len(spr["train"]), "train examples.")


# ------------- vocab -------------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id, unk_id = vocab["<PAD>"], vocab["<UNK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ------------- encode -------------
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ------------- collate -------------
def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in input_ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(input_ids):
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
    for split in ["train", "dev", "test"]
}


# ------------- model -------------
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


# ------------- train / eval helpers -------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ------------- hyperparameter tuning -------------
nhead_values = [2, 4, 8]
epochs = 10

for nhead in nhead_values:
    print(f"\n==== Training with nhead={nhead} ====")
    model = CharTransformer(vocab_size, num_labels, nhead=nhead).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best_f1, best_state = 0.0, None
    history = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": []},
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"], criterion)
        history["epochs"].append(epoch)
        history["losses"]["train"].append(tr_loss)
        history["losses"]["val"].append(val_loss)
        history["metrics"]["train_f1"].append(tr_f1)
        history["metrics"]["val_f1"].append(val_f1)
        print(
            f"nhead={nhead} | epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    # --- test with best checkpoint ---
    model.load_state_dict(best_state)
    test_loss, test_f1, test_preds, test_gts = run_epoch(
        model, loaders["test"], criterion
    )
    print(f"nhead={nhead} | BEST val_f1 {best_f1:.4f} | Test f1 {test_f1:.4f}")

    # --- save results ---
    experiment_data["nhead_tuning"]["SPR_BENCH"]["results"][nhead] = {
        "history": history,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "predictions": test_preds,
        "ground_truth": test_gts,
    }
    # free memory
    del model
    torch.cuda.empty_cache()

# ------------- persist -------------
np.save("experiment_data.npy", experiment_data)
print("Saved experiment data to experiment_data.npy")
