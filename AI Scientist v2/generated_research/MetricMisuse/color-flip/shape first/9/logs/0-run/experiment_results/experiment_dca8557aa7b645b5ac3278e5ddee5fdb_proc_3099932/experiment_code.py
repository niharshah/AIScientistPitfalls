# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ---------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- load SPR --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset, DatasetDict as HFDD

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = HFDD()
    dset["train"], dset["dev"], dset["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return dset


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------------- vocab build -----------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["train"]["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr)
print("Vocab size:", len(vocab))

# --------------- label mapping ----------------
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels)
print("Num classes:", num_classes)


# --------------- Dataset class ---------------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, seq):
        return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        return self.encode_seq(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate(batch):
    seqs, labels, raw = zip(*batch)
    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    padded = np.full((len(seqs), maxlen), vocab["<pad>"], dtype=np.int64)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return (torch.tensor(padded), torch.tensor(lengths), torch.tensor(labels)), raw


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# --------------- model -----------------------
class MeanEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, nclass):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, nclass)
        )

    def forward(self, x, lengths):
        emb = self.emb(x)  # B,L,D
        mask = (x != 0).unsqueeze(-1)  # B,L,1
        mean = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # B,D
        return self.classifier(mean)


# --------------- experiment data -------------
experiment_data = {
    "num_epochs_sweep": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},  # epoch-wise for best run
            "losses": {"train": [], "val": []},  # epoch-wise for best run
            "predictions": [],
            "ground_truth": [],
            "config_epochs": [],  # list of tried max_epochs
            "best_val_metric": [],  # best CWA-2D per config
        }
    }
}


# --------------- training function -----------
def train_model(max_epochs=20, patience=3, emb_dim=64, lr=1e-3, tol=1e-4):
    model = MeanEncoder(len(vocab), emb_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss, best_state, best_metrics, best_epoch = float("inf"), None, None, 0
    train_losses, val_losses = [], []
    for epoch in range(1, max_epochs + 1):
        # -------- train ----------
        model.train()
        epoch_loss = 0.0
        for (x, lens, y), _ in train_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x, lens), y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * x.size(0)
        train_loss = epoch_loss / len(train_loader.dataset)
        # -------- val ------------
        model.eval()
        vloss, preds, gtruth, seqs = 0.0, [], [], []
        with torch.no_grad():
            for (x, lens, y), raw in dev_loader:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                out = model(x, lens)
                loss = criterion(out, y)
                vloss += loss.item() * x.size(0)
                preds.extend(out.argmax(1).cpu().tolist())
                gtruth.extend(y.cpu().tolist())
                seqs.extend(raw)
        vloss /= len(dev_loader.dataset)
        cwa = complexity_weighted_accuracy(
            seqs, [id2label[i] for i in gtruth], [id2label[i] for i in preds]
        )
        train_losses.append(train_loss)
        val_losses.append(vloss)
        print(
            f"[max_epochs={max_epochs}] Epoch {epoch}: val_loss={vloss:.4f} CWA-2D={cwa:.4f}"
        )
        # early stopping bookkeeping
        if vloss < best_val_loss - tol:
            best_val_loss, best_state, best_metrics, best_epoch = (
                vloss,
                model.state_dict(),
                (train_losses[:], val_losses[:], preds[:], gtruth[:], cwa),
                epoch,
            )
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    # restore best
    model.load_state_dict(best_state)
    return best_epoch, best_metrics


# --------------- sweep over num_epochs -------
epoch_grid = [10, 20, 30, 40, 50]
best_run_idx, best_cwa = None, -1.0
for idx, max_ep in enumerate(epoch_grid):
    wait = 0  # reset global var for each run
    best_epoch, bm = train_model(max_epochs=max_ep)
    tr_losses, v_losses, preds, gtruth, cwa = bm
    experiment_data["num_epochs_sweep"]["SPR_BENCH"]["config_epochs"].append(max_ep)
    experiment_data["num_epochs_sweep"]["SPR_BENCH"]["best_val_metric"].append(cwa)
    if cwa > best_cwa:
        best_cwa, best_run_idx = cwa, idx
        # keep curves & predictions of best run
        experiment_data["num_epochs_sweep"]["SPR_BENCH"]["losses"]["train"] = tr_losses
        experiment_data["num_epochs_sweep"]["SPR_BENCH"]["losses"]["val"] = v_losses
        experiment_data["num_epochs_sweep"]["SPR_BENCH"]["predictions"] = preds
        experiment_data["num_epochs_sweep"]["SPR_BENCH"]["ground_truth"] = gtruth

print(
    f"Best run: grid index {best_run_idx}, max_epochs={epoch_grid[best_run_idx]}, best CWA-2D={best_cwa:.4f}"
)

# --------------- save experiment data --------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --------------- plot (best run) -------------
plt.figure()
plt.plot(
    experiment_data["num_epochs_sweep"]["SPR_BENCH"]["losses"]["train"], label="train"
)
plt.plot(experiment_data["num_epochs_sweep"]["SPR_BENCH"]["losses"]["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Best run loss curves")
plt.legend()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()

print("Finished. Data & plot saved in ./working/")
