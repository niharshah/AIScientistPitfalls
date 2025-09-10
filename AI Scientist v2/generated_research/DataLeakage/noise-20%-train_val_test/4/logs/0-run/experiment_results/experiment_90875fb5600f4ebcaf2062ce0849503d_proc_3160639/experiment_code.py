import os, pathlib, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict

# ---------- experiment bookkeeping ----------
experiment_data = {"d_model_tuning": {"SPR_BENCH": {}}}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

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
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------- collate ----------
def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in input_ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(input_ids):
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
        self, vocab_size, num_labels, d_model=128, nhead=4, nlayers=4, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(256, d_model * 2),  # scale feed-forward with size
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]  # (B,L,D)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, split, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds, gts = 0.0, [], []
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
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- hyperparameter sweep ----------
d_model_values = [96, 128, 192, 256, 384]
epochs = 10

for dm in d_model_values:
    print(f"\n=== Training with d_model={dm} ===")
    model = CharTransformer(vocab_size, num_labels, d_model=dm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # prepare storage
    experiment_data["d_model_tuning"]["SPR_BENCH"][dm] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, epochs + 1)),
    }

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, loaders["train"], "train", optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["dev"], "dev")

        # log
        edata = experiment_data["d_model_tuning"]["SPR_BENCH"][dm]
        edata["losses"]["train"].append(tr_loss)
        edata["losses"]["val"].append(val_loss)
        edata["metrics"]["train"].append(tr_f1)
        edata["metrics"]["val"].append(val_f1)

        print(
            f"d_model {dm} | Epoch {epoch:2d}: "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    # -------- test with best checkpoint --------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_f1, test_preds, test_gts = run_epoch(model, loaders["test"], "test")
    print(f"d_model {dm} | Test MacroF1: {test_f1:.4f}")

    # save predictions
    edata = experiment_data["d_model_tuning"]["SPR_BENCH"][dm]
    edata["predictions"] = test_preds
    edata["ground_truth"] = test_gts
    edata["best_val_f1"] = best_val_f1
    edata["test_f1"] = test_f1

    # free GPU memory before next run
    del model
    torch.cuda.empty_cache()

# ---------- persist results ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments completed and saved to experiment_data.npy")
