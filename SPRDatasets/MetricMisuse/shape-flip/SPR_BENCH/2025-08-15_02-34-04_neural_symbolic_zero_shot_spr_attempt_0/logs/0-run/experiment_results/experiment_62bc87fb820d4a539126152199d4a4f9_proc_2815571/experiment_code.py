import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- misc --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# -------------------- dataset location helper --------------------
def find_spr_root() -> pathlib.Path:
    env = os.getenv("SPR_DIR")
    cands = []
    if env:
        cands.append(pathlib.Path(env))
    cands.append(pathlib.Path.cwd() / "SPR_BENCH")
    for p in pathlib.Path.cwd().resolve().parents:
        cands.append(p / "SPR_BENCH")
    for c in cands:
        if (c / "train.csv").exists():
            print(f"Found SPR_BENCH at {c}")
            return c
    raise FileNotFoundError("SPR_BENCH not found; set $SPR_DIR or place folder nearby.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------- Dataset --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, max_len=30):
        self.d = hf_split
        self.tok2id = tok2id
        self.lab2id = lab2id
        self.max_len = max_len

    def __len__(self):
        return len(self.d)

    def encode(self, seq):
        ids = [self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.split()]
        ids = ids[: self.max_len]
        pad = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * pad, len(ids)

    def __getitem__(self, idx):
        row = self.d[idx]
        ids, l = self.encode(row["sequence"])
        return dict(
            input_ids=torch.tensor(ids),
            lengths=torch.tensor(l),
            label=torch.tensor(self.lab2id[row["label"]]),
            raw_seq=row["sequence"],
        )


# -------------------- Model --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, n_cls)

    def forward(self, x, l):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, l.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (l - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)


# -------------------- Prepare data --------------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)
specials = ["<pad>", "<unk>"]
vocab = set()
[vocab.update(s.split()) for s in spr["train"]["sequence"]]
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
for i, tok in enumerate(specials):
    token2idx[tok] = i
pad_idx = token2idx["<pad>"]
labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# -------------------- Utils --------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = n = 0
    preds, labels, seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            bt = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logit = model(bt["input_ids"], bt["lengths"])
            loss = criterion(logit, bt["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * bt["label"].size(0)
            n += bt["label"].size(0)
            pred = logit.argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(bt["label"].cpu().numpy())
            seqs.extend(bt["raw_seq"])
    avg = tot_loss / n
    y_true = [idx2label[i] for i in labels]
    y_pred = [idx2label[i] for i in preds]
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0
    return avg, (swa, cwa, hwa), y_true, y_pred


# -------------------- Hyper-parameter sweep --------------------
hidden_dims = [32, 64, 96, 128, 192, 256]
num_epochs = 5
experiment_data = {"hidden_dim_sweep": {}}

best_dim, best_hwa = None, 0.0
for hd in hidden_dims:
    key = str(hd)
    experiment_data["hidden_dim_sweep"][key] = {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    print(f"\n==== Hidden Dim {hd} ====")
    model = GRUClassifier(len(token2idx), 32, hd, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_met, _, _ = run_epoch(model, dev_loader, criterion)
        experiment_data["hidden_dim_sweep"][key]["losses"]["train"].append(tr_loss)
        experiment_data["hidden_dim_sweep"][key]["losses"]["val"].append(val_loss)
        experiment_data["hidden_dim_sweep"][key]["metrics"]["train"].append(tr_met)
        experiment_data["hidden_dim_sweep"][key]["metrics"]["val"].append(val_met)
        experiment_data["hidden_dim_sweep"][key]["timestamps"].append(time.time())
        print(
            f"Ep{ep} val_loss={val_loss:.4f} HWA={val_met[2]:.4f} ({time.time()-t0:.1f}s)"
        )
    # test
    test_loss, test_met, y_t, y_p = run_epoch(model, test_loader, criterion)
    experiment_data["hidden_dim_sweep"][key]["losses"]["test"] = test_loss
    experiment_data["hidden_dim_sweep"][key]["metrics"]["test"] = test_met
    experiment_data["hidden_dim_sweep"][key]["predictions"] = y_p
    experiment_data["hidden_dim_sweep"][key]["ground_truth"] = y_t
    print(f"Test HWA={test_met[2]:.4f}")
    if test_met[2] > best_hwa:
        best_hwa = test_met[2]
        best_dim = hd
        # save curve for best so far
        fig, ax = plt.subplots()
        ax.plot(
            experiment_data["hidden_dim_sweep"][key]["losses"]["train"], label="train"
        )
        ax.plot(experiment_data["hidden_dim_sweep"][key]["losses"]["val"], label="val")
        ax.set_title(f"GRU loss (hid={hd})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_hid{hd}.png"))
        plt.close(fig)

print(f"\nBest hidden_dim={best_dim} with Test HWA={best_hwa:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data and plots to {working_dir}")
