import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from typing import List, Dict

# ---------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- data loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


default_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not default_path.exists():
    default_path = pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(default_path)
print({k: len(v) for k, v in spr.items()})


# ---------- vocabulary ----------
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


# ---------- dataset ----------
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
    lengths = torch.tensor([len(x) for x in inputs])
    max_len = max(lengths)
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, : len(seq)] = seq
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lengths, "label": labels}


batch_size = 128
train_dl = DataLoader(
    SPRTorchDataset(spr["train"], vocab),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_dl = DataLoader(
    SPRTorchDataset(spr["dev"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_dl = DataLoader(
    SPRTorchDataset(spr["test"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

num_classes = len(set(spr["train"]["label"]))
print("Classes:", num_classes)


# ---------- model ----------
class CharBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim, num_cls, drop_p: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(hid_dim * 2, num_cls)

    def forward(self, x, lengths):
        emb = self.embedding(x.to(device))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        mask = (x != 0).unsqueeze(-1).to(out.dtype)
        mean = (out * mask).sum(1) / lengths.unsqueeze(1).to(out.dtype)
        mean = self.dropout(mean)
        return self.fc(mean)


# ---------- helpers ----------
def run_epoch(model, dl, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds_all, labels_all = 0.0, [], []
    for batch in dl:
        # move tensors to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(batch["label"].cpu().numpy())
    avg_loss = tot_loss / len(dl.dataset)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    return avg_loss, macro_f1, np.array(preds_all), np.array(labels_all)


# ---------- experiment ----------
dropout_grid = [0.0, 0.3, 0.5]
epochs = 12
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": {}, "val_f1": {}, "test_f1": {}},
        "losses": {"train": {}, "val": {}},
        "predictions": {},
        "ground_truth": {},
    }
}

val_f1_curves = {}

for p in dropout_grid:
    print(f"\n=== Training with dropout={p} ===")
    model = CharBiLSTM(vocab_size, 64, 160, num_classes, p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_dev, best_state = 0.0, None
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"][p] = []
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"][p] = []
    experiment_data["SPR_BENCH"]["losses"]["train"][p] = []
    experiment_data["SPR_BENCH"]["losses"]["val"][p] = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_f1={tr_f1:.4f} val_f1={val_f1:.4f}"
        )
        experiment_data["SPR_BENCH"]["metrics"]["train_f1"][p].append(tr_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val_f1"][p].append(val_f1)
        experiment_data["SPR_BENCH"]["losses"]["train"][p].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"][p].append(val_loss)
        if val_f1 > best_dev:
            best_dev, best_state = val_f1, model.state_dict()
    # test with best
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, test_dl, criterion)
    experiment_data["SPR_BENCH"]["metrics"]["test_f1"][p] = test_f1
    experiment_data["SPR_BENCH"]["predictions"][p] = preds.tolist()
    experiment_data["SPR_BENCH"]["ground_truth"][p] = gts.tolist()
    print(f"Best Dev F1={best_dev:.4f} | Test F1={test_f1:.4f}")
    val_f1_curves[p] = experiment_data["SPR_BENCH"]["metrics"]["val_f1"][p]

# ---------- save experiment data ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# ---------- plot ----------
plt.figure()
for p, curve in val_f1_curves.items():
    plt.plot(curve, label=f"dropout {p}")
plt.xlabel("Epoch")
plt.ylabel("Val Macro-F1")
plt.title("Validation Macro-F1 vs Epochs")
plt.legend()
plt.savefig(os.path.join(working_dir, "SPR_val_f1_dropout_curves.png"))
plt.close()
