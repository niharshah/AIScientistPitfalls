import os, pathlib, math, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# ---------- misc ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {"num_layers_tuning": {"SPR_BENCH": {}}}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- dataset loader ----------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocabulary ----------
special_tokens = ["<PAD>"]
chars = set(ch for seq in spr["train"]["sequence"] for ch in seq)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size: {vocab_size}  Num classes: {num_classes}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[ch] for ch in self.seqs[idx]], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "label": label}


def collate_fn(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    return {
        "input_ids": pad_sequence(seqs, batch_first=True, padding_value=pad_id),
        "label": labels,
    }


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, nclass, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, nclass)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


# ---------- helper ----------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


# ---------- hyperparameter sweep ----------
layer_choices = [
    1,
    3,
    4,
    6,
]  # 2 is baseline but we include implicitly later for comparison
for nlayers in layer_choices + [2]:  # ensures 2 is also evaluated last
    print(f"\n=== Training model with num_layers={nlayers} ===")
    model = SimpleTransformer(
        vocab_size,
        d_model=128,
        nhead=4,
        nlayers=nlayers,
        nclass=num_classes,
        pad_idx=pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # storage
    layer_key = str(nlayers)
    experiment_data["num_layers_tuning"]["SPR_BENCH"][layer_key] = {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},  # kept for potential extension
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            optimizer.zero_grad()
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["label"].size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss, val_f1, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"Layer={nlayers}  Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        ed = experiment_data["num_layers_tuning"]["SPR_BENCH"][layer_key]
        ed["metrics"]["train_loss"].append(train_loss)
        ed["metrics"]["val_loss"].append(val_loss)
        ed["metrics"]["val_f1"].append(val_f1)
        ed["epochs"].append(epoch)

    # final test evaluation
    test_loss, test_f1, preds, gts = evaluate(model, test_loader, criterion)
    print(f"Layer={nlayers}  TEST: loss={test_loss:.4f}  macro_f1={test_f1:.4f}")
    ed["losses"]["train"] = ed["metrics"]["train_loss"]
    ed["losses"]["val"] = ed["metrics"]["val_loss"]
    ed["predictions"] = preds
    ed["ground_truth"] = gts
    ed["test_loss"] = test_loss
    ed["test_f1"] = test_f1

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
