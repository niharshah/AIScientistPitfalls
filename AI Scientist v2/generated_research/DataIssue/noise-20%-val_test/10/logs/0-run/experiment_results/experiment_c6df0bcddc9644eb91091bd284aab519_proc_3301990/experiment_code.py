import os, pathlib, json, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from typing import List, Dict
import copy, random

# ---------------- reproducibility -----------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- data loading -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for sp in ["train", "dev", "test"]:
        dset[sp] = _load(f"{sp}.csv")
    return dset


default_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not default_path.exists():
    default_path = pathlib.Path("SPR_BENCH/")
spr = load_spr_bench(default_path)
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


# ---------------- dataset and dataloader -----------------
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
        # mean pooling over valid timesteps
        mask = (x != 0).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lens = lengths.unsqueeze(1).to(out.dtype)
        mean = summed / lens
        return self.fc(mean)


# ---------------- helpers -----------------
def run_epoch(model, dl, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().numpy())
    avg_loss = total_loss / len(dl.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels)


# ---------------- hyper-parameter tuning over embedding_dim -----------------
embedding_dims = [32, 64, 128, 256]
epochs = 5
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"embedding_dim": {"SPR_BENCH": {}}}

for emb_dim in embedding_dims:
    print(f"\n===== Training with embedding_dim={emb_dim} =====")
    model = CharBiLSTM(vocab_size, emb_dim, hidden_dim=128, num_classes=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # store per-epoch history
    hist = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
    }

    best_state, best_val_f1 = None, 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, dev_dl, criterion)
        hist["metrics"]["train_f1"].append(tr_f1)
        hist["metrics"]["val_f1"].append(val_f1)
        hist["losses"]["train"].append(tr_loss)
        hist["losses"]["val"].append(val_loss)
        print(
            f"Epoch {epoch}/{epochs} | train_loss {tr_loss:.4f} val_loss {val_loss:.4f} "
            f"train_f1 {tr_f1:.4f} val_f1 {val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    # test with best model
    model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = run_epoch(model, test_dl, criterion)
    print(f"Best Dev F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}")

    # save experiment info
    dim_key = f"emb_{emb_dim}"
    exp_dict = copy.deepcopy(hist)
    exp_dict["predictions"] = preds.tolist()
    exp_dict["ground_truth"] = gts.tolist()
    exp_dict["test_f1"] = test_f1
    experiment_data["embedding_dim"]["SPR_BENCH"][dim_key] = exp_dict

    # plot val f1 curve
    plt.figure()
    plt.plot(hist["metrics"]["val_f1"], label=f"val_f1 (emb={emb_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.title(f"Validation Macro-F1 (emb_dim={emb_dim})")
    plt.savefig(os.path.join(working_dir, f"spr_val_f1_emb{emb_dim}.png"))
    plt.close()
    torch.cuda.empty_cache()

# summarize best val f1 for each dim
dims, bests = [], []
for emb_dim in embedding_dims:
    key = f"emb_{emb_dim}"
    dims.append(emb_dim)
    bests.append(
        max(experiment_data["embedding_dim"]["SPR_BENCH"][key]["metrics"]["val_f1"])
    )
plt.figure()
plt.plot(dims, bests, marker="o")
plt.xlabel("Embedding Dim")
plt.ylabel("Best Val Macro-F1")
plt.title("Embedding Dim vs Best Val Macro-F1")
plt.savefig(os.path.join(working_dir, "spr_val_f1_vs_embdim.png"))
plt.close()

# ---------------- save all data -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data at", os.path.join(working_dir, "experiment_data.npy"))
