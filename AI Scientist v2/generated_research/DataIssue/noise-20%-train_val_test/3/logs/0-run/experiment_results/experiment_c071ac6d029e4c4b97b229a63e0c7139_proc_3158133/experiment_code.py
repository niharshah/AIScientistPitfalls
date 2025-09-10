# -------------------------- hyper-parameter sweep: learning-rate --------------------------
import os, pathlib, random, math, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# reproducibility --------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# experiment bookkeeping -------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"learning_rate": {}}  # hierarchy: sweep_type -> lr_key -> data

# device -----------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------- dataset -------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"], dset["dev"], dset["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return dset


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# vocabulary --------------------------------------------------------------------------------
special_tokens = ["<PAD>"]
chars = set("".join(spr["train"]["sequence"]))
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id = stoi["<PAD>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# torch dataset ----------------------------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[ch] for ch in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    padded = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": padded, "label": labels}


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr["train"]),
    SPRTorchDataset(spr["dev"]),
    SPRTorchDataset(spr["test"]),
)
batch_size = 128
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# model ------------------------------------------------------------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, num_layers, n_classes, pad):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, pad_mask):
        x = self.embed(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).type_as(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


# evaluation -------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pad_mask = batch["input_ids"] == pad_id
        logits = model(batch["input_ids"], pad_mask)
        loss = criterion(logits, batch["label"])
        tot_loss += loss.item() * batch["label"].size(0)
        preds += logits.argmax(-1).cpu().tolist()
        gts += batch["label"].cpu().tolist()
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# -------------------------- learning rate sweep -------------------------------------------
lr_candidates = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
epochs = 5
d_model = 128

for lr in lr_candidates:
    lr_key = f"lr_{lr:.0e}"
    print(f"\n===== training with learning_rate={lr} =====")
    model = SimpleTransformer(vocab_size, d_model, 4, 2, num_classes, pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    exp_entry = {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "lr": lr,
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pad_mask = batch["input_ids"] == pad_id
            logits = model(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["label"].size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        exp_entry["metrics"]["train_loss"].append(train_loss)
        exp_entry["metrics"]["val_loss"].append(val_loss)
        exp_entry["metrics"]["val_f1"].append(val_f1)
        exp_entry["epochs"].append(epoch)

    # final test evaluation ----------------------------------------------------------------
    test_loss, test_f1, test_preds, test_gts = evaluate(model, test_loader)
    print(f"Test: loss={test_loss:.4f} macro_f1={test_f1:.4f}")
    exp_entry["test_loss"], exp_entry["test_f1"] = test_loss, test_f1
    exp_entry["predictions"], exp_entry["ground_truth"] = test_preds, test_gts

    experiment_data["learning_rate"][lr_key] = exp_entry
    # cleanup GPU memory before next sweep value
    del model
    torch.cuda.empty_cache()

# -------------------------- persist everything --------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
