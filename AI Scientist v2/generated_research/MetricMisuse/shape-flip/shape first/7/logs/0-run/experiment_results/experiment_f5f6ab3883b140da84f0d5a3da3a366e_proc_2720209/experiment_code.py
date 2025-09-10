import os, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# Working directory & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Book-keeping dict
experiment_data = {
    "SPR_Hybrid": {
        "metrics": {"train_SWA": [], "val_SWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
# Metric helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1e-8)


# ------------------------------------------------------------------
# Dataset loading (with synthetic fallback)
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def make_synthetic_dataset(path: str, n_train=3000, n_dev=1000, n_test=1000):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # simplistic label rule
        return int(count_shape_variety(seq) >= count_color_variety(seq))

    def write(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

    os.makedirs(path, exist_ok=True)
    write(n_train, "train.csv")
    write(n_dev, "dev.csv")
    write(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found – creating synthetic data …")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------------
# Vocabulary & encoding
def build_vocab(hf_dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in hf_dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
# Torch Dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        sym = torch.tensor(
            [
                count_shape_variety(seq) / 10.0,
                count_color_variety(seq) / 10.0,
                len(seq.split()) / 10.0,
            ],
            dtype=torch.float,
        )
        return {
            "input": torch.tensor(encode(seq), dtype=torch.long),
            "sym": sym,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# ------------------------------------------------------------------
# Hybrid Model
class HybridGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hid=128, sym_dim=3, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.sym_fc = nn.Linear(sym_dim, hid * 2)
        self.classifier = nn.Linear(hid * 2, num_classes)

    def forward(self, x, sym):
        h = self.emb(x)  # [B,L,E]
        h, _ = self.gru(h)  # [B,L,2H]
        h = h.mean(dim=1)  # pooled
        s = torch.relu(self.sym_fc(sym))  # [B,2H]
        h = h + s  # fuse neural + symbolic
        return self.classifier(h)


# ------------------------------------------------------------------
# Training / evaluation utilities
def run_epoch(model, dataloader, criterion, optimiser=None):
    train_mode = optimiser is not None
    model.train() if train_mode else model.eval()
    total_loss, y_true, y_pred, raw_seqs = 0.0, [], [], []
    for batch in dataloader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["sym"])
        loss = criterion(logits, batch["label"])
        if train_mode:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
        raw_seqs.extend(batch["raw"])
    avg_loss = total_loss / len(dataloader.dataset)
    swa = shape_weighted_accuracy(raw_seqs, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# ------------------------------------------------------------------
# Instantiate model
model = HybridGRU(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 8

# ------------------------------------------------------------------
# Training loop
for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(model, train_dl, criterion, optimizer)
    val_loss, val_swa, _, _ = run_epoch(model, dev_dl, criterion)
    experiment_data["SPR_Hybrid"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_Hybrid"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_Hybrid"]["metrics"]["train_SWA"].append((epoch, tr_swa))
    experiment_data["SPR_Hybrid"]["metrics"]["val_SWA"].append((epoch, val_swa))
    print(f"Epoch {epoch}: val_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")

# ------------------------------------------------------------------
# Final test evaluation
_, test_swa, gt, preds = run_epoch(model, test_dl, criterion)
print(f"\nTest Shape-Weighted Accuracy (SWA): {test_swa:.4f}")
experiment_data["SPR_Hybrid"]["predictions"] = preds
experiment_data["SPR_Hybrid"]["ground_truth"] = gt

# ------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
