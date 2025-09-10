import os, random, torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------
# Working dir / device setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------
# Experiment-wide store
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------
# Utility metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


# -----------------------------------------------------------
# Data loading (real or synthetic fallback)
def load_spr_bench(root: str) -> DatasetDict:
    def _load(name):  # treat each csv as single split
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def make_synthetic_dataset(path: str, n_tr=5000, n_dev=1000, n_te=1000):
    shapes, colors = list("STCH"), list("RGBY")

    def rnd_seq():
        L = random.randint(3, 10)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):  # toy rule: label 1 if #shapes > #colors else 0
        return int(count_shape_variety(seq) > len(set(t[1] for t in seq.split())))

    os.makedirs(path, exist_ok=True)
    for n, split in [(n_tr, "train"), (n_dev, "dev"), (n_te, "test")]:
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rnd_seq()
                f.write(f"{i},{s},{rule(s)}\n")


root = "SPR_BENCH"
if not (
    os.path.isdir(root)
    and all(
        os.path.isfile(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("SPR_BENCH not found; creating synthetic placeholder.")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# -----------------------------------------------------------
# Tokeniser & vocab
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(tok, 1) for tok in seq.split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# -----------------------------------------------------------
# PyTorch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input": torch.tensor(encode(seq), dtype=torch.long),
            "sym": torch.tensor(
                [count_shape_variety(seq), len(set(tok[1] for tok in seq.split()))],
                dtype=torch.float,
            ),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": seq,
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# -----------------------------------------------------------
# Neural-symbolic model
class HybridGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hid=128, sym_dim=2, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.sym_fc = nn.Linear(sym_dim, 16)
        self.out = nn.Linear(hid * 2 + 16, n_cls)

    def forward(self, tok_ids, sym_feats):
        x = self.emb(tok_ids)
        gru_out, _ = self.gru(x)
        pooled = gru_out.mean(1)
        sym_emb = torch.relu(self.sym_fc(sym_feats))
        concat = torch.cat([pooled, sym_emb], dim=-1)
        return self.out(concat)


model = HybridGRU(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -----------------------------------------------------------
# Training / evaluation loop
def run_epoch(dl, train=False):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, y_true, y_pred, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            # move tensors
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch_t["input"], batch_t["sym"])
            loss = criterion(logits, batch_t["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(batch_t["label"])
            preds = logits.argmax(-1).detach().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch_t["label"].cpu().tolist())
            seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, y_true, y_pred


# -----------------------------------------------------------
epochs = 8
for epoch in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_swa, _, _ = run_epoch(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((epoch, tr_swa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((epoch, val_swa))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")

# -----------------------------------------------------------
# Test evaluation
_, test_swa, gt, pred = run_epoch(test_dl)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = pred
experiment_data["SPR_BENCH"]["ground_truth"] = gt

# -----------------------------------------------------------
# Save experiment record
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "Saved metrics and predictions to", os.path.join(working_dir, "experiment_data.npy")
)
