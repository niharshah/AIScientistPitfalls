import os, pathlib, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset

# ---------------- working dir -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- load SPR --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) else 0.0


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


# --------------- Dataset class ----------------
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
    lens = [len(s) for s in seqs]
    maxlen = max(lens)
    padded = np.full((len(seqs), maxlen), vocab["<pad>"], dtype=np.int64)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return (torch.tensor(padded), torch.tensor(lens), torch.tensor(labels)), raw


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# --------------- model definition -------------
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


# --------------- hyper-parameter grid ---------
weight_decays = [0.0, 1e-5, 5e-5, 1e-4, 5e-4]
num_epochs = 5

# --------------- experiment data dict ---------
experiment_data = {
    "weight_decay": {
        str(wd): {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for wd in weight_decays
    }
}

# --------------- training loop ---------------
criterion = nn.CrossEntropyLoss()
val_loss_curves = {}  # for plotting

for wd in weight_decays:
    key = str(wd)
    print(f"\n=== Training with weight_decay={wd} ===")
    model = MeanEncoder(len(vocab), 64, num_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
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
        experiment_data["weight_decay"][key]["losses"]["train"].append(train_loss)

        # ---- validate ----
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
        cwa2d = complexity_weighted_accuracy(
            all_seq, [id2label[i] for i in all_true], [id2label[i] for i in all_pred]
        )
        experiment_data["weight_decay"][key]["losses"]["val"].append(val_loss)
        experiment_data["weight_decay"][key]["metrics"]["val"].append(cwa2d)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f} | CWA-2D={cwa2d:.4f}")

    # store last epoch predictions/ground truth
    experiment_data["weight_decay"][key]["predictions"] = all_pred
    experiment_data["weight_decay"][key]["ground_truth"] = all_true
    val_loss_curves[key] = experiment_data["weight_decay"][key]["losses"]["val"]

# --------------- save experiment data ---------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --------------- plot val loss curves ---------
plt.figure()
for wd_key, losses in val_loss_curves.items():
    plt.plot(losses, label=f"wd={wd_key}")
plt.title("Validation Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()

print("Finished. Data & plot saved in ./working/")
