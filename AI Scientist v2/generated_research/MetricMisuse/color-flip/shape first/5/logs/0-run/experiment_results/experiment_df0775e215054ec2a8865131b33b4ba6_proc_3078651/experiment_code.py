# NoContrastivePretraining – single-file ablation
import os, random, math, pathlib, time, itertools, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# Try to load SPR_BENCH, else build synthetic toy data
def safe_load_spr():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("SPR_BENCH")
        if DATA_PATH.exists():
            return load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Could not load real SPR_BENCH; falling back to synthetic.", e)

    shapes, colors = list("ABCDEFG"), list("123456")

    def random_token():
        return random.choice(shapes) + random.choice(colors)

    def random_seq():
        return " ".join(random_token() for _ in range(random.randint(4, 10)))

    def label_fn(seq):
        return 1 if sum(tok[0] in "ABC" for tok in seq.split()) % 2 == 0 else 0

    synthetic = {"train": [], "dev": [], "test": []}
    for split, n in [("train", 2000), ("dev", 400), ("test", 400)]:
        for i in range(n):
            seq = random_seq()
            synthetic[split].append({"id": i, "sequence": seq, "label": label_fn(seq)})
    return synthetic


dset = safe_load_spr()
print({k: len(v) for k, v in dset.items()})


# ------------------------------------------------------------
# metrics
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def cwca(seqs, y_true, y_pred):
    weights = [(count_shape_variety(s) + count_color_variety(s)) / 2 for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ------------------------------------------------------------
# vocab
PAD, MASK, UNK = "<PAD>", "<MASK>", "<UNK>"


def build_vocab(seqs):
    vocab = set(itertools.chain.from_iterable(s.split() for s in seqs))
    vocab = [PAD, MASK, UNK] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


vocab, stoi = build_vocab([ex["sequence"] for ex in dset["train"]])
itos = {i: s for s, i in stoi.items()}


def encode(seq: str) -> List[int]:
    return [stoi.get(tok, stoi[UNK]) for tok in seq.split()]


# ------------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, recs, with_label=True):
        self.recs = recs
        self.with_label = with_label

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rec = self.recs[idx]
        item = {"input_ids": torch.tensor(encode(rec["sequence"]), dtype=torch.long)}
        if self.with_label:
            item["label"] = torch.tensor(rec["label"], dtype=torch.long)
        return item


def collate_classification(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return {
        "input_ids": padded.to(device),
        "label": labels.to(device),
        "lengths": torch.tensor(lengths).to(device),
    }


# ------------------------------------------------------------
# model
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, lens):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        h_mean = (h * mask).sum(1) / (mask.sum(1) + 1e-6)
        return self.linear(h_mean)


class SPRModel(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x, lens):
        rep = self.encoder(x, lens)
        return self.classifier(rep), rep


# ------------------------------------------------------------
# data loaders
train_loader = DataLoader(
    SPRDataset(dset["train"]),
    batch_size=64,
    shuffle=True,
    collate_fn=collate_classification,
)
dev_loader = DataLoader(
    SPRDataset(dset["dev"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)
test_loader = DataLoader(
    SPRDataset(dset["test"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)

# ------------------------------------------------------------
# experiment logging dict
experiment_data = {
    "NoContrastivePretraining": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------------------------------------------------
# initialise model (random) – no contrastive pre-training
model = SPRModel(len(vocab), num_labels=len(set(r["label"] for r in dset["train"]))).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
clf_epochs = 5


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, preds, trues, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            logits, _ = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            pred = logits.argmax(1).detach().cpu().numpy()
            true = batch["label"].detach().cpu().numpy()
            preds.extend(pred)
            trues.extend(true)
            seqs.extend(
                [
                    " ".join(itos[idx.item()] for idx in row if idx.item() != 0)
                    for row in batch["input_ids"].cpu()
                ]
            )
    cwca_val = cwca(seqs, trues, preds)
    return total_loss / len(loader.dataset), cwca_val, preds, trues


best_val = 0
for epoch in range(1, clf_epochs + 1):
    tr_loss, tr_cwca, _, _ = run_epoch(train_loader, True)
    val_loss, val_cwca, _, _ = run_epoch(dev_loader, False)
    experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["losses"]["train"].append(
        tr_loss
    )
    experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["losses"]["val"].append(
        val_loss
    )
    experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["metrics"]["train"].append(
        tr_cwca
    )
    experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["metrics"]["val"].append(
        val_cwca
    )
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_CWCA={val_cwca:.4f}"
    )
    best_val = max(best_val, val_cwca)

# ------------------------------------------------------------
# final test evaluation
test_loss, test_cwca, preds, trues = run_epoch(test_loader, False)
print(f"Test CWCA = {test_cwca:.4f}")
experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["metrics"]["test"].append(
    test_cwca
)
experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["predictions"] = preds
experiment_data["NoContrastivePretraining"]["SPR_BENCH"]["ground_truth"] = trues

# ------------------------------------------------------------
# t-SNE visualisation on dev embeddings
model.eval()
embs, labels = [], []
with torch.no_grad():
    for batch in dev_loader:
        _, reps = model(batch["input_ids"], batch["lengths"])
        embs.append(reps.cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
embs = np.concatenate(embs, 0)
tsne = TSNE(n_components=2, init="random", perplexity=30, random_state=0).fit_transform(
    embs
)
plt.figure(figsize=(6, 5))
plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap="tab10", s=10)
plt.title("t-SNE of dev embeddings (NoContrastivePretraining)")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "tsne_dev.png"))

# ------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data and plot to ./working/")
