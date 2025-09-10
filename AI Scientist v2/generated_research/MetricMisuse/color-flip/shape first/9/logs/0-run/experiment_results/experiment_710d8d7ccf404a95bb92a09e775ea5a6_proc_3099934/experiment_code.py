import os, pathlib, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def count_shape_variety(sequence: str):
    return len({tok[0] for tok in sequence.strip().split() if tok})


def count_color_variety(sequence: str):
    return len({tok[1] for tok in sequence.strip().split() if len(tok) > 1})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

# ---------- vocab ----------------
vocab = {"<pad>": 0, "<unk>": 1}
for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        vocab.setdefault(tok, len(vocab))
print("Vocab size:", len(vocab))

# ---------- label maps -----------
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels)


# ---------- dataset --------------
class SPRTorchDataset(Dataset):
    def __init__(self, dsplit):
        self.seqs = dsplit["sequence"]
        self.labels = [label2id[l] for l in dsplit["label"]]

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [vocab.get(tok, 1) for tok in seq.strip().split()]

    def __getitem__(self, idx):
        return self.encode(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate(batch):
    seqs, labels, raw = zip(*batch)
    lens = [len(s) for s in seqs]
    mlen = max(lens)
    pad_arr = np.full((len(seqs), mlen), 0, np.int64)
    for i, s in enumerate(seqs):
        pad_arr[i, : len(s)] = s
    return (torch.tensor(pad_arr), torch.tensor(lens), torch.tensor(labels)), raw


train_ds, dev_ds = SPRTorchDataset(spr["train"]), SPRTorchDataset(spr["dev"])


# ---------- model ----------------
class MeanEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, nclass):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.clf = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, nclass)
        )

    def forward(self, x, lens):
        e = self.emb(x)
        mask = (x != 0).unsqueeze(-1)
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.clf(mean)


# ---------- experiment store -----
experiment_data = {"batch_size": {"SPR_BENCH": {}}}  # each bs will populate here


# ---------- training routine -----
def run_experiment(batch_size, num_epochs=5, emb_dim=64, lr=1e-3):
    print(f"\n=== Running batch_size={batch_size} ===")
    tr_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    dv_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = MeanEncoder(len(vocab), emb_dim, num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    losses_tr, losses_val, metrics_val = [], [], []
    best_pred, best_true = None, None

    for ep in range(1, num_epochs + 1):
        model.train()
        tot = 0
        for (x, lens, y), _ in tr_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            optim.zero_grad()
            out = model(x, lens)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            tot += loss.item() * x.size(0)
        tr_loss = tot / len(train_ds)
        losses_tr.append(tr_loss)

        # ------ validation -----
        model.eval()
        tot = 0
        preds, truth, seqs = [], [], []
        with torch.no_grad():
            for (x, lens, y), raw in dv_loader:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                out = model(x, lens)
                loss = crit(out, y)
                tot += loss.item() * x.size(0)
                pr = out.argmax(1).cpu().tolist()
                preds.extend(pr)
                truth.extend(y.cpu().tolist())
                seqs.extend(raw)
        vl_loss = tot / len(dev_ds)
        losses_val.append(vl_loss)
        cwa = complexity_weighted_accuracy(
            seqs, [id2label[i] for i in truth], [id2label[i] for i in preds]
        )
        metrics_val.append(cwa)
        print(
            f"Epoch {ep}: train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  CWA={cwa:.4f}"
        )
        best_pred, best_true = (
            preds,
            truth,
        )  # overwrite; after last epoch they correspond to final model

    # ------ store ---------------
    experiment_data["batch_size"]["SPR_BENCH"][batch_size] = {
        "metrics": {"train": [], "val": metrics_val},
        "losses": {"train": losses_tr, "val": losses_val},
        "predictions": best_pred,
        "ground_truth": best_true,
    }

    # ------ plot -----------------
    plt.figure()
    plt.plot(losses_tr, label="train")
    plt.plot(losses_val, label="val")
    plt.title(f"Loss (bs={batch_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_loss_bs{batch_size}.png"))
    plt.close()


# ---------- sweep ---------------
for bs in [32, 64, 128, 256]:
    run_experiment(bs)

# ---------- save ---------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll runs finished. Results saved in ./working/")
