import os, pathlib, json, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from typing import List, Dict

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


root_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
root_path = root_path if root_path.exists() else pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(root_path)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocabulary -----------------
def build_vocab(dataset) -> Dict[str, int]:
    charset = set()
    for seq in dataset["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(spr["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------------- dataset -----------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab):
        self.seq = split["sequence"]
        self.labels = split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq)

    def encode(self, s):
        return [self.vocab[c] for c in s]

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [b["input"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor([len(x) for x in inputs])
    max_len = lengths.max().item()
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, : len(seq)] = seq
    return {"input": padded, "lengths": lengths, "label": labels}


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"], vocab), batch_size, True, collate_fn=collate_fn
)
dev_dl = DataLoader(
    SPRTorchDataset(spr["dev"], vocab), batch_size, False, collate_fn=collate_fn
)
test_dl = DataLoader(
    SPRTorchDataset(spr["test"], vocab), batch_size, False, collate_fn=collate_fn
)

num_classes = len(set(spr["train"]["label"]))
print("Classes:", num_classes)


# ---------------- model -----------------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        summed = (out * mask).sum(1)
        mean = summed / lengths.unsqueeze(1).to(out.dtype)
        return self.fc(mean)


# ---------------- helpers -----------------
def run_epoch(model, dl, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, labels = 0.0, [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    avg_loss = tot_loss / len(dl.dataset)
    f1 = f1_score(labels, preds, average="macro")
    return avg_loss, f1, np.array(preds), np.array(labels)


# ---------------- hyperparameter sweep -----------------
hidden_dims = [64, 128, 192, 256]
epochs = 10
working_dir = "working"
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"hidden_dim": {"SPR_BENCH": {}}}
best_overall_f1, best_preds, best_gts = 0.0, None, None

for hd in hidden_dims:
    print(f"\n===== Training with hidden_dim={hd} =====")
    model = CharBiLSTM(
        vocab_size, emb_dim=64, hidden_dim=hd, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_store = {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": []},
        "losses": {"train": [], "val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
    }
    best_val_f1 = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, *_ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, *_ = run_epoch(model, dev_dl, criterion)
        run_store["losses"]["train"].append(tr_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["train_f1"].append(tr_f1)
        run_store["metrics"]["val_f1"].append(val_f1)
        print(
            f"Ep{ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"tr_f1={tr_f1:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    # Evaluate best on test
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, test_dl, criterion)
    run_store["losses"]["test"].append(test_loss)
    run_store["metrics"]["test_f1"].append(test_f1)
    run_store["predictions"] = preds.tolist()
    run_store["ground_truth"] = gts.tolist()
    print(f"hidden_dim={hd} Test Macro-F1: {test_f1:.4f}")

    experiment_data["hidden_dim"]["SPR_BENCH"][str(hd)] = run_store

    # track best overall
    if test_f1 > best_overall_f1:
        best_overall_f1 = test_f1
        best_preds, best_gts = preds, gts

# ---------------- save metrics -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------------- visualization -----------------
vals = [
    experiment_data["hidden_dim"]["SPR_BENCH"][str(h)]["metrics"]["val_f1"][-1]
    for h in hidden_dims
]
plt.figure()
plt.plot(hidden_dims, vals, marker="o")
plt.xlabel("hidden_dim")
plt.ylabel("Best Dev Macro-F1")
plt.title("Hidden Dim Tuning")
plt.savefig(os.path.join(working_dir, "hidden_dim_tuning.png"))
plt.close()
