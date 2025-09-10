import os, pathlib, time, json, math, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- misc setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------- locate SPR --------------------
def find_spr_root() -> pathlib.Path:
    cand = os.getenv("SPR_DIR")
    if cand and (pathlib.Path(cand) / "train.csv").exists():
        return pathlib.Path(cand)
    p = pathlib.Path.cwd()
    for c in [p / "SPR_BENCH"] + [par / "SPR_BENCH" for par in p.resolve().parents]:
        if (c / "train.csv").exists():
            return c
    raise FileNotFoundError("SPR_BENCH not found; set $SPR_DIR or place folder.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


# -------------------- metrics helpers --------------------
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------- Dataset --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, max_len=30):
        self.data, self.tok2id, self.lab2id, self.max_len = (
            hf_split,
            tok2id,
            lab2id,
            max_len,
        )

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        ids = [
            self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.strip().split()
        ][: self.max_len]
        return ids + [self.tok2id["<pad>"]] * (self.max_len - len(ids)), len(ids)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, l = self.encode(row["sequence"])
        return {
            "input_ids": torch.tensor(ids),
            "lengths": torch.tensor(l),
            "label": torch.tensor(self.lab2id[row["label"]]),
            "raw_seq": row["sequence"],
        }


# -------------------- Model --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, n_cls)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)


# ----------------------------------------------------------------
# Prepare shared vocabulary/labels (same for all batch sizes)
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


# -------------------- training helpers --------------------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot = 0
    preds, labels_, seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(bt["input_ids"], bt["lengths"])
            loss = criterion(logits, bt["label"])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * bt["label"].size(0)
            tot += bt["label"].size(0)
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p)
            labels_.extend(bt["label"].cpu().numpy())
            seqs.extend(bt["raw_seq"])
    avg = tot_loss / tot
    y_true = [idx2label[i] for i in labels_]
    y_pred = [idx2label[i] for i in preds]
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg, (swa, cwa, hwa), y_true, y_pred


# -------------------- hyperparameter tuning loop --------------------
batch_sizes = [64, 128, 512, 1024]
num_epochs = 5
experiment_data = {"batch_size": {}}

for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    # model & optimisation
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # storage
    key = f"spr_bench_bs{bs}"
    experiment_data["batch_size"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    # epochs
    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_met, _, _ = run_epoch(model, dev_loader, criterion, None)
        ed = experiment_data["batch_size"][key]
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train"].append(tr_met)
        ed["metrics"]["val"].append(val_met)
        ed["timestamps"].append(time.time())
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f}  SWA={val_met[0]:.4f} "
            f"CWA={val_met[1]:.4f}  HWA={val_met[2]:.4f}  ({time.time()-t0:.1f}s)"
        )
    # final test
    test_loss, test_met, y_tst, y_pred = run_epoch(model, test_loader, criterion, None)
    ed["losses"]["test"] = test_loss
    ed["metrics"]["test"] = test_met
    ed["predictions"] = y_pred
    ed["ground_truth"] = y_tst
    print(
        f"Test -> SWA={test_met[0]:.4f}  CWA={test_met[1]:.4f}  HWA={test_met[2]:.4f}"
    )

    # plot loss curve
    plt.figure()
    plt.plot(ed["losses"]["train"], label="train")
    plt.plot(ed["losses"]["val"], label="val")
    plt.title(f"Loss (bs={bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"spr_loss_bs{bs}.png"))
    plt.close()

# -------------------- save all experiments --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll outputs saved to {working_dir}")
