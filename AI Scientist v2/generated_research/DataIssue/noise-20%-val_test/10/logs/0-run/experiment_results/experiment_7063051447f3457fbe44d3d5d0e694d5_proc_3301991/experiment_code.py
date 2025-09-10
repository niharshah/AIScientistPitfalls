import os, pathlib, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import List, Dict

# ---------------- experiment store -----------------
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- data loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
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


root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not root.exists():
    root = pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocabulary -----------------
def build_vocab(dataset) -> Dict[str, int]:
    charset = set()
    for seq in dataset["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(list(charset)))}  # 0 = PAD
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(spr["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------------- dataset class -----------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seq = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq)

    def encode(self, s: str) -> List[int]:
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
    max_len = max(lengths)
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, : len(seq)] = seq
    return {"input": padded, "lengths": lengths, "label": labels}


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
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().numpy())
    avg_loss = total_loss / len(dl.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


# ---------------- training per batch size -----------------
batch_sizes = [32, 64, 128, 256]
epochs = 10
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    train_dl = DataLoader(
        SPRTorchDataset(spr["train"], vocab),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_dl = DataLoader(
        SPRTorchDataset(spr["dev"], vocab),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        SPRTorchDataset(spr["test"], vocab),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = CharBiLSTM(
        vocab_size, emb_dim=64, hidden_dim=128, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp_key = f"bs_{bs}"
    experiment_data["batch_size_tuning"]["SPR_BENCH"][exp_key] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    best_dev_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
        info = experiment_data["batch_size_tuning"]["SPR_BENCH"][exp_key]
        info["losses"]["train"].append(tr_loss)
        info["losses"]["val"].append(val_loss)
        info["metrics"]["train_f1"].append(tr_f1)
        info["metrics"]["val_f1"].append(val_f1)
        print(
            f"Epoch {epoch}: loss {tr_loss:.4f}/{val_loss:.4f} "
            f"f1 {tr_f1:.4f}/{val_f1:.4f}"
        )
        if val_f1 > best_dev_f1:
            best_dev_f1 = val_f1
            best_state = model.state_dict()

    # ---- evaluation on test with best model ----
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, test_dl, criterion)
    print(f"Best Dev F1={best_dev_f1:.4f} | Test F1={test_f1:.4f}")

    info = experiment_data["batch_size_tuning"]["SPR_BENCH"][exp_key]
    info["predictions"] = preds.tolist()
    info["ground_truth"] = gts.tolist()
    info["test_f1"] = test_f1
    info["best_dev_f1"] = best_dev_f1

    # save per-bs curve
    plt.figure()
    plt.plot(info["metrics"]["val_f1"], label=f"Val F1 (bs={bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.title(f"Validation Macro-F1 (batch={bs})")
    plt.savefig(os.path.join(working_dir, f"val_f1_bs{bs}.png"))
    plt.close()

# ---------------- save all experiment data -----------------
np.save("experiment_data.npy", experiment_data, allow_pickle=True)
print("All experiment data saved to experiment_data.npy")
