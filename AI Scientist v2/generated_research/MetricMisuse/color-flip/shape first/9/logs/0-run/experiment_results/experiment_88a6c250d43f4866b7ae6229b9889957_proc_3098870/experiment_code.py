import os, pathlib, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ---------- experiment bookkeeping ----------
experiment_data = {
    "embedding_dim": {  # hyper-parameter tuning type
        # every run will be inserted as  "SPR_BENCH_dim{d}" : {...}
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR -------------
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
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------- utils -----------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["train"]["sequence"]:
        for tok in seq.strip().split():
            vocab.setdefault(tok, len(vocab))
    return vocab


def count_shape_variety(sequence: str):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(cor) / sum(w) if sum(w) else 0.0


vocab = build_vocab(spr)
print("Vocab size:", len(vocab))

labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels)
print("Num classes:", num_classes)


# ---------- dataset --------------
class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.raw_seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.raw_seqs)

    def encode_seq(self, seq):
        return [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        return self.encode_seq(self.raw_seqs[idx]), self.labels[idx], self.raw_seqs[idx]


def collate(batch):
    seqs, labels, raw = zip(*batch)
    lengths = [len(s) for s in seqs]
    maxlen = max(lengths)
    pad = vocab["<pad>"]
    padded = np.full((len(batch), maxlen), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return (torch.tensor(padded), torch.tensor(lengths), torch.tensor(labels)), raw


train_ds, dev_ds = SPRTorchDataset(spr["train"]), SPRTorchDataset(spr["dev"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- model ----------------
class MeanEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, nclass):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.clf = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, nclass)
        )

    def forward(self, x, lengths):
        e = self.emb(x)  # B,L,D
        mask = (x != 0).unsqueeze(-1)  # B,L,1
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.clf(mean)


# ---------- hyper-parameter loop --
embed_dims = [32, 64, 128, 256]
num_epochs = 5

for d in embed_dims:
    tag = f"SPR_BENCH_dim{d}"
    print(f"\n=== Training embed_dim={d} ===")
    run_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = MeanEncoder(len(vocab), d, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        for (x, lens, y), _ in train_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optim.zero_grad()
            out = model(x, lens)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * x.size(0)
        train_loss = epoch_loss / len(train_ds)
        run_dict["losses"]["train"].append(train_loss)

        # ---- val ----
        model.eval()
        v_loss, preds, truths, raws = 0.0, [], [], []
        with torch.no_grad():
            for (x, lens, y), raw in dev_loader:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                out = model(x, lens)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                preds.extend(out.argmax(1).cpu().tolist())
                truths.extend(y.cpu().tolist())
                raws.extend(raw)
        v_loss /= len(dev_ds)
        cwa = complexity_weighted_accuracy(
            raws, [id2label[i] for i in truths], [id2label[i] for i in preds]
        )
        run_dict["losses"]["val"].append(v_loss)
        run_dict["metrics"]["val"].append(cwa)
        print(
            f"  Epoch {epoch}: train_loss={train_loss:.4f} val_loss={v_loss:.4f} CWA-2D={cwa:.4f}"
        )

    # save last predictions / g.t.
    run_dict["predictions"] = preds
    run_dict["ground_truth"] = truths
    experiment_data["embedding_dim"][tag] = run_dict

    # ---- plot ----
    plt.figure()
    plt.plot(run_dict["losses"]["train"], label="train")
    plt.plot(run_dict["losses"]["val"], label="val")
    plt.title(f"Loss curves (emb_dim={d})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{tag}_loss_curve.png"))
    plt.close()

# ---------- persist everything ----
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nFinished. All data & plots saved in ./working/")
