import os, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from typing import List, Dict
from datasets import DatasetDict

# ------------------ working directory ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ device -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ load SPR_BENCH ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


possible_roots = [
    pathlib.Path(os.getenv("SPR_DIR", "SPR_BENCH")),  # env var or cwd
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),  # default from prompt
]
spr_path = None
for p in possible_roots:
    if p.exists():
        spr_path = p
        break
if spr_path is None:
    raise FileNotFoundError("Could not locate SPR_BENCH directory.")

spr_bench = load_spr_bench(spr_path)
print("Loaded SPR_BENCH splits:", spr_bench.keys())


# ------------------ vocab & label maps -----------------
def build_vocab(seqs: List[str]) -> Dict[str, int]:
    chars = set("".join(seqs))
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 reserved for PAD
    return vocab


train_seqs = spr_bench["train"]["sequence"]
vocab = build_vocab(train_seqs)
pad_idx = 0

labels = sorted(list(set(spr_bench["train"]["label"])))
label2id = {lab: i for i, lab in enumerate(labels)}

num_classes = len(labels)
print(f"Vocab size: {len(vocab)}, Num classes: {num_classes}")


# ------------------ torch Dataset ----------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2id):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, s: str):
        return [self.vocab[c] for c in s]

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        lab = self.labels[idx]
        return {
            "input_ids": torch.tensor(self.encode_seq(seq), dtype=torch.long),
            "labels": torch.tensor(self.label2id[lab], dtype=torch.long),
        }


def collate_fn(batch):
    # dynamic pad
    lengths = [len(item["input_ids"]) for item in batch]
    max_len = max(lengths)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
    return {"input_ids": input_ids, "labels": labels}


# ------------------ model ------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_idx)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # [B,L,D]
        mask = (input_ids != pad_idx).unsqueeze(-1).float()  # [B,L,1]
        summed = (emb * mask).sum(1)  # [B,D]
        lengths = mask.sum(1)  # [B,1]
        avg = summed / torch.clamp(lengths, min=1.0)  # [B,D]
        logits = self.classifier(avg)
        return logits


# ------------------ datasets & loaders -----------------
train_ds = SPRTorchDataset(spr_bench["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr_bench["dev"], vocab, label2id)
test_ds = SPRTorchDataset(spr_bench["test"], vocab, label2id)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

# ------------------ training setup ---------------------
model = AvgEmbedClassifier(
    len(vocab), embed_dim=32, num_classes=num_classes, pad_idx=pad_idx
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# experiment data store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


def evaluate(loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


best_dev_f1, best_state = 0.0, None
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_f1 = None  # computing F1 on full train set is expensive, skip or quick eval
    val_loss, val_f1, _, _ = evaluate(dev_loader)

    # store metrics
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macro_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_macro_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | validation_loss = {val_loss:.4f} | val_macro_F1 = {val_f1:.4f}"
    )
    if val_f1 > best_dev_f1:
        best_dev_f1 = val_f1
        best_state = model.state_dict()

# ------------------ final test evaluation --------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_labels = evaluate(test_loader)
print(f"Best Dev Macro-F1: {best_dev_f1:.4f} | Test Macro-F1: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels

# save embedding weights for interpretability
np.save(
    os.path.join(working_dir, "char_embeddings.npy"),
    model.embedding.weight.detach().cpu().numpy(),
)

# ------------------ persist experiment data ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
