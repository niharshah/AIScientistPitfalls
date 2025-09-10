import os, pathlib, json, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset
from typing import List, Dict

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- data loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not root.exists():
    root = pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocabulary -----------------
def build_vocab(dataset) -> Dict[str, int]:
    charset = {c for seq in dataset["sequence"] for c in seq}
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
        self.seq, self.labels, self.vocab = split["sequence"], split["label"], vocab

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        encoded = torch.tensor([self.vocab[c] for c in self.seq[idx]], dtype=torch.long)
        return {
            "input": encoded,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    seqs = [b["input"] for b in batch]
    lens = torch.tensor([len(s) for s in seqs])
    max_len = lens.max().item()
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "label": labels}


def make_loader(split, shuffle):
    return DataLoader(
        SPRTorchDataset(split, vocab),
        batch_size=128,
        shuffle=shuffle,
        collate_fn=collate,
    )


train_dl, dev_dl, test_dl = map(
    make_loader, (spr["train"], spr["dev"], spr["test"]), (True, False, False)
)
num_classes = len(set(spr["train"]["label"]))
print("Classes:", num_classes)


# ---------------- model -----------------
class CharBiLSTM(nn.Module):
    def __init__(self):  # fixed dims
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths.unsqueeze(1).to(out.dtype)
        return self.fc(mean)


# ---------------- helpers -----------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, preds, labels = 0.0, [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    return (
        tot / len(loader.dataset),
        f1_score(labels, preds, average="macro"),
        np.array(preds),
        np.array(labels),
    )


# ---------------- hyperparameter tuning: num_epochs -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {}
epoch_choices = [10, 30, 50]
patience = 6

for epochs in epoch_choices:
    tag = f"num_epochs_{epochs}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = CharBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1, wait = 0.0, 0
    best_path = os.path.join(working_dir, f"best_{tag}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
        ed = experiment_data[tag]["SPR_BENCH"]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_f1"].append(tr_f1)
        ed["metrics"]["val_f1"].append(val_f1)
        print(
            f"[{tag}] Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} train_f1={tr_f1:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # --- load best and evaluate on test ---
    model.load_state_dict(torch.load(best_path))
    _, _, preds, gts = run_epoch(model, test_dl, criterion)
    ed["predictions"] = preds.tolist()
    ed["ground_truth"] = gts.tolist()
    print(f"[{tag}] Best Dev F1: {best_f1:.4f}")

    # optional plot
    plt.figure()
    plt.plot(ed["metrics"]["val_f1"])
    plt.title(f"{tag} Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.savefig(os.path.join(working_dir, f"{tag}_val_f1.png"))
    plt.close()

# ---------------- save all experiments -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All experiments saved.")
