import os, pathlib, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ------------------------- working dir --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------- device -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- seed ---------------------------------
torch.manual_seed(42)
np.random.seed(42)


# ------------------------- load SPR -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset, DatasetDict as HFDD

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = HFDD()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ------------------------- helpers ------------------------------
def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["train"]["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr)
print("Vocab size:", len(vocab))

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels)
print("Num classes:", num_classes)


# ------------------------ dataset class -------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return self.encode(seq), self.labels[idx], seq


def collate(batch):
    seqs, labels, raw = zip(*batch)
    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    padded = np.full((len(seqs), maxlen), vocab["<pad>"], dtype=np.int64)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return (torch.tensor(padded), torch.tensor(lengths), torch.tensor(labels)), raw


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])


# ------------------------ model ---------------------------------
class MeanEncoder(nn.Module):
    def __init__(self, vsz, edim, ncls):
        super().__init__()
        self.emb = nn.Embedding(vsz, edim, padding_idx=0)
        self.clf = nn.Sequential(nn.Linear(edim, 128), nn.ReLU(), nn.Linear(128, ncls))

    def forward(self, x, lengths):
        emb = self.emb(x)
        mask = (x != 0).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        mean = summed / denom
        return self.clf(mean)


# -------------------- hyperparameter sweep ----------------------
epoch_options = [5, 10, 20, 30]
experiment_data = {}

for max_epochs in epoch_options:
    tag = f"num_epochs_{max_epochs}"
    experiment_data[tag] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    # fresh data loaders each run (same objects ok)
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = MeanEncoder(len(vocab), 64, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, max_epochs + 1):
        # -------- training --------
        model.train()
        running_loss = 0.0
        for (x, lens, y), _ in train_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optimiser.zero_grad()
            out = model(x, lens)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)
        experiment_data[tag]["SPR_BENCH"]["losses"]["train"].append(train_loss)

        # -------- validation --------
        model.eval()
        vloss, all_pred, all_true, all_seq = 0.0, [], [], []
        with torch.no_grad():
            for (x, lens, y), raw in dev_loader:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                out = model(x, lens)
                loss = criterion(out, y)
                vloss += loss.item() * x.size(0)
                preds = out.argmax(1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(y.cpu().tolist())
                all_seq.extend(raw)
        vloss /= len(dev_ds)
        cwa = complexity_weighted_accuracy(
            all_seq,
            [id2label[i] for i in all_true],
            [id2label[i] for i in all_pred],
        )
        experiment_data[tag]["SPR_BENCH"]["losses"]["val"].append(vloss)
        experiment_data[tag]["SPR_BENCH"]["metrics"]["val"].append(cwa)

        print(
            f"[{tag}] Epoch {epoch}/{max_epochs} | val_loss={vloss:.4f} | CWA-2D={cwa:.4f}"
        )

    # store final predictions / ground truth
    experiment_data[tag]["SPR_BENCH"]["predictions"] = all_pred
    experiment_data[tag]["SPR_BENCH"]["ground_truth"] = all_true

    # ----------- plot -------------
    plt.figure()
    plt.plot(experiment_data[tag]["SPR_BENCH"]["losses"]["train"], label="train")
    plt.plot(experiment_data[tag]["SPR_BENCH"]["losses"]["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves ({tag})")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{tag}_loss_curve.png"))
    plt.close()

# ----------------------- save all data --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. All data and plots saved in ./working/")
