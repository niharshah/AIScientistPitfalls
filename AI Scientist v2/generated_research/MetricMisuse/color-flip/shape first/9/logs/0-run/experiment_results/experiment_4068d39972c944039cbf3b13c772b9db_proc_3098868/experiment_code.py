import os, pathlib, time, random, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset

# ---------------- reproducibility ------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- SPR helpers ----------------
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


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


# ---------------- load data ------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------------- vocab / label maps ----------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["train"]["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels)
print(f"Vocab size: {len(vocab)} | Num classes: {num_classes}")


# ---------------- dataset objects ------------
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


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------------- model ----------------------
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
        summed = (emb * mask).sum(1)  # B,D
        lens = mask.sum(1).clamp(min=1)  # B,1
        mean = summed / lens  # B,D
        return self.classifier(mean)


# ---------------- hyperparameter sweep -------
learning_rates = [3e-4, 5e-4, 1e-3, 2e-3]
num_epochs = 5

experiment_data = {"learning_rate": {"SPR_BENCH": {}}}

for lr in learning_rates:
    print(f"\n==== Training with learning rate = {lr} ====")
    model = MeanEncoder(len(vocab), 64, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    run_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, num_epochs + 1):
        # --------- training -------------
        model.train()
        epoch_loss = 0.0
        for (x, lens, y), _ in train_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optimiser.zero_grad()
            out = model(x, lens)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * x.size(0)
        train_loss = epoch_loss / len(train_ds)
        run_record["losses"]["train"].append(train_loss)

        # --------- validation ----------
        model.eval()
        val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
        with torch.no_grad():
            for (x, lens, y), raw in dev_loader:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                out = model(x, lens)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(y.cpu().tolist())
                all_seq.extend(raw)
        val_loss /= len(dev_ds)
        run_record["losses"]["val"].append(val_loss)

        cwa2d = complexity_weighted_accuracy(
            all_seq, [id2label[i] for i in all_true], [id2label[i] for i in all_pred]
        )
        run_record["metrics"]["val"].append(cwa2d)
        print(
            f"Epoch {epoch}/{num_epochs} | TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | CWA-2D {cwa2d:.4f}"
        )

    # Store last epoch predictions/ground_truth
    run_record["predictions"] = all_pred
    run_record["ground_truth"] = all_true
    experiment_data["learning_rate"]["SPR_BENCH"][f"lr_{lr}"] = run_record

    # --------- plot losses for this lr -------
    plt.figure()
    plt.plot(run_record["losses"]["train"], label="train")
    plt.plot(run_record["losses"]["val"], label="val")
    plt.title(f"Loss curves (lr={lr})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_loss_curve_lr_{lr}.png"))
    plt.close()

# -------------- save all experiment data -----
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nFinished. All data & plots saved in ./working/")
