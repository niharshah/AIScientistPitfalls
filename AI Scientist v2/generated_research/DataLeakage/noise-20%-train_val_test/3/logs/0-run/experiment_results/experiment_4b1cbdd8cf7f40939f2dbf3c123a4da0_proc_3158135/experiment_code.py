import os, pathlib, math, random, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------- experiment dict --------------------------
experiment_data = {
    "nhead": {
        "SPR_BENCH": {
            "values": [],  # tried nhead values
            "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------- device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------- dataset --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------------- vocab --------------------------
special_tokens = ["<PAD>"]
chars = {ch for seq in spr["train"]["sequence"] for ch in seq}
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, vocab_size = stoi["<PAD>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
print("Vocab", vocab_size, "Classes", num_classes)


# -------------------------- torch dataset --------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size, True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size, False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size, False, collate_fn=collate_fn
)


# -------------------------- model --------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        h = self.embed(x)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(h)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


# -------------------------- train / eval helpers --------------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["input_ids"] == pad_id)
        loss = criterion(logits, batch["label"])
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    return avg_loss, f1_score(gts, preds, average="macro"), preds, gts


# -------------------------- hyperparameter sweep --------------------------
d_model, num_layers, epochs = 128, 2, 5
nhead_options = [2, 4, 8]
best_val_f1, best_state = -1.0, None

for nhead in nhead_options:
    print(f"\n=== Training with nhead={nhead} ===")
    model = SimpleTransformer(
        vocab_size, d_model, nhead, num_layers, num_classes, pad_id
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["input_ids"] == pad_id)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch["label"].size(0)
        train_loss = ep_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
    # log metrics (use last epoch values)
    experiment_data["nhead"]["SPR_BENCH"]["values"].append(nhead)
    experiment_data["nhead"]["SPR_BENCH"]["metrics"]["train_loss"].append(train_loss)
    experiment_data["nhead"]["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["nhead"]["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    # keep best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_nhead = nhead

# -------------------------- test evaluation with best model --------------------------
print(f"\nBest nhead={best_nhead} with dev F1={best_val_f1:.4f}")
best_model = SimpleTransformer(
    vocab_size, d_model, best_nhead, num_layers, num_classes, pad_id
).to(device)
best_model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = evaluate(best_model, test_loader)
print(f"Test : loss={test_loss:.4f} macro_f1={test_f1:.4f}")

experiment_data["nhead"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["nhead"]["SPR_BENCH"]["ground_truth"] = test_gts

# -------------------------- save --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
