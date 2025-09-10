import os, pathlib, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR-BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_path = spr_path if spr_path.exists() else pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(spr_path)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab ----------
def build_vocab(ds):
    charset = set()
    for seq in ds["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(spr["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab):
        self.seqs, self.labels, self.vocab = split["sequence"], split["label"], vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_enc = [self.vocab[c] for c in self.seqs[idx]]
        return {
            "input": torch.tensor(seq_enc, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    seqs = [b["input"] for b in batch]
    lens = torch.tensor([len(s) for s in seqs])
    maxlen = lens.max()
    padded = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "label": labels}


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"], vocab), batch_size, True, collate_fn=collate
)
dev_dl = DataLoader(
    SPRTorchDataset(spr["dev"], vocab), batch_size, False, collate_fn=collate
)
test_dl = DataLoader(
    SPRTorchDataset(spr["test"], vocab), batch_size, False, collate_fn=collate
)
num_classes = len(set(spr["train"]["label"]))
print("Classes:", num_classes)


# ---------- model ----------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, n_cls)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        summed = (out * mask).sum(1)
        mean = summed / lengths.unsqueeze(1).to(out.dtype)
        return self.fc(mean)


# ---------- helpers ----------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dl, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss, preds, labels = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    return (
        tot_loss / len(dl.dataset),
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


# ---------- experiment store ----------
experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "runs": {},  # metrics per lr
            "best_lr": None,
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- hyperparameter tuning ----------
lrs = [5e-4, 7e-4, 2e-3]
epochs = 35
best_overall_f1, best_lr, best_state = 0.0, None, None

for lr in lrs:
    print(f"\n=== Training with lr={lr} ===")
    model = CharBiLSTM(vocab_size, 64, 128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_store = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
    }
    best_dev_f1 = 0.0
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl)
        run_store["losses"]["train"].append(tr_loss)
        run_store["losses"]["val"].append(val_loss)
        run_store["metrics"]["train_f1"].append(tr_f1)
        run_store["metrics"]["val_f1"].append(val_f1)
        if val_f1 > best_dev_f1:
            best_dev_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(working_dir, "tmp_best.pt"))
        if ep % 5 == 0 or ep == 1:
            print(
                f"Ep{ep}: tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} tr_f1={tr_f1:.4f} val_f1={val_f1:.4f}"
            )
    # store run info
    experiment_data["learning_rate"]["SPR_BENCH"]["runs"][str(lr)] = run_store
    # check global best
    if best_dev_f1 > best_overall_f1:
        best_overall_f1, best_lr = best_dev_f1, lr
        best_state = torch.load(os.path.join(working_dir, "tmp_best.pt"))

print(f"\nBest lr={best_lr} with dev Macro-F1={best_overall_f1:.4f}")
experiment_data["learning_rate"]["SPR_BENCH"]["best_lr"] = best_lr

# ---------- test evaluation ----------
best_model = CharBiLSTM(vocab_size, 64, 128, num_classes).to(device)
best_model.load_state_dict(best_state)
test_loss, test_f1, preds, gts = run_epoch(best_model, test_dl)
print(f"Test Macro_F1_Score: {test_f1:.4f}")

experiment_data["learning_rate"]["SPR_BENCH"]["predictions"] = preds
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = gts

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- plot ----------
plt.figure()
for lr in lrs:
    plt.plot(
        experiment_data["learning_rate"]["SPR_BENCH"]["runs"][str(lr)]["metrics"][
            "val_f1"
        ],
        label=f"lr={lr}",
    )
plt.xlabel("Epoch")
plt.ylabel("Val Macro-F1")
plt.title("Validation Macro-F1 across LRs")
plt.legend()
plt.savefig(os.path.join(working_dir, "spr_val_f1_curve.png"))
plt.close()
