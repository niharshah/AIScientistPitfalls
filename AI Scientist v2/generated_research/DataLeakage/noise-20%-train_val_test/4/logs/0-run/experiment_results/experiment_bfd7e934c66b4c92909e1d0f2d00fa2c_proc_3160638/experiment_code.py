import os, pathlib, math, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict

# ---------------- experiment data container ----------------
experiment_data = {
    "num_epochs": {  # hyperparameter tuning type
        "SPR_BENCH": {
            "tried_epochs": [],
            "epoch_curves": {  # epoch-wise lists per variant
                "train_loss": [],
                "val_loss": [],
                "train_f1": [],
                "val_f1": [],
            },
            "best_val_f1": [],
            "test_f1": [],
            "predictions": [],
            "ground_truth": [],  # filled once
        }
    }
}

# ---------------- reproducibility ----------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- dataset loader ----------------
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


# ---------------- build vocab ----------------
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


# ---------------- encode sequences ----------------
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------------- collate fn ----------------
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


# ---------------- model ----------------
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


# ---------------- helper: one epoch ----------------
def run_epoch(model, split, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    loader = loaders[split]
    criterion = nn.CrossEntropyLoss()
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


# ---------------- hyperparameter sweep ----------------
max_epochs_grid = [10, 20, 30, 40, 50]
patience = 5  # early stopping patience
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

for max_epochs in max_epochs_grid:
    print(f"\n==== Training with max_epochs = {max_epochs} ====")
    experiment_data["num_epochs"]["SPR_BENCH"]["tried_epochs"].append(max_epochs)

    model = CharTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val_f1, best_state, waited = 0.0, None, 0
    curve_train_loss, curve_val_loss, curve_train_f1, curve_val_f1 = [], [], [], []

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, "train", optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, "dev")
        curve_train_loss.append(tr_loss)
        curve_val_loss.append(val_loss)
        curve_train_f1.append(tr_f1)
        curve_val_f1.append(val_f1)
        print(
            f"Epoch {epoch}/{max_epochs} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                print("Early stopping triggered.")
                break

    # save epoch curves
    for k, lst in zip(
        ["train_loss", "val_loss", "train_f1", "val_f1"],
        [curve_train_loss, curve_val_loss, curve_train_f1, curve_val_f1],
    ):
        experiment_data["num_epochs"]["SPR_BENCH"]["epoch_curves"][k].append(lst)

    experiment_data["num_epochs"]["SPR_BENCH"]["best_val_f1"].append(best_val_f1)

    # ---------------- test evaluation with best ckpt ----------------
    model.load_state_dict(best_state)
    test_loss, test_f1, test_preds, test_gts = run_epoch(model, "test")
    print(f"Test MacroF1 (best ckpt): {test_f1:.4f}")

    experiment_data["num_epochs"]["SPR_BENCH"]["test_f1"].append(test_f1)
    experiment_data["num_epochs"]["SPR_BENCH"]["predictions"].append(test_preds)
    if not experiment_data["num_epochs"]["SPR_BENCH"]["ground_truth"]:
        experiment_data["num_epochs"]["SPR_BENCH"]["ground_truth"] = test_gts

# ---------------- save all results ----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
