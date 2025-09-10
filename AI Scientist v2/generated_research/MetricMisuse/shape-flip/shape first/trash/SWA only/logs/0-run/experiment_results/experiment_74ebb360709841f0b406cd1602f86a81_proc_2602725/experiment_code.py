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

import os, random, string, datetime, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------ house-keeping -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"weight_decay": {}}  # hyper-parameter tuning container

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------- data -----------------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{s}.csv")) for s in ["train", "dev", "test"]
    )


use_synthetic = not spr_files_exist(SPR_PATH)

if use_synthetic:
    print("Real SPR_BENCH not found – generating synthetic data.")
    shapes = list(string.ascii_uppercase[:6])
    colors = [str(i) for i in range(4)]
    randseq = lambda: " ".join(
        random.choice(shapes) + random.choice(colors)
        for _ in range(random.randint(4, 9))
    )
    rule = lambda seq: int(
        len(set(t[0] for t in seq.split())) == len(set(t[1] for t in seq.split()))
    )

    def make(n):
        seq = [randseq() for _ in range(n)]
        lab = [rule(s) for s in seq]
        return {"sequence": seq, "label": lab}

    raw_data = {"train": make(2000), "dev": make(400), "test": make(600)}
else:
    print("Loading real SPR_BENCH")
    from datasets import load_dataset, DatasetDict

    def _load(split):
        return load_dataset(
            "csv",
            data_files=os.path.join(SPR_PATH, f"{split}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict()
    ds["train"], ds["dev"], ds["test"] = [_load(s) for s in ["train", "dev", "test"]]
    raw_data = {
        s: {"sequence": ds[s]["sequence"], "label": ds[s]["label"]}
        for s in ["train", "dev", "test"]
    }

# ----------------------- helpers --------------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vocab_size = len(vocab)


def enc(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRTorchDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = [torch.tensor(enc(s), dtype=torch.long) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw_seq = seqs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "label": self.y[idx]}


def collate(batch):
    lens = [len(x["input_ids"]) for x in batch]
    maxlen = max(lens)
    inp = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    lab = torch.empty(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        inp[i, : len(b["input_ids"])] = b["input_ids"]
        lab[i] = b["label"]
    return {"input_ids": inp, "labels": lab, "lengths": torch.tensor(lens)}


datasets = {
    s: SPRTorchDataset(raw_data[s]["sequence"], raw_data[s]["label"])
    for s in ["train", "dev", "test"]
}


# accuracy helpers
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split()))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split()))


def shape_weighted(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


def color_weighted(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)) / (sum(w) or 1)


def signatures(seqs):
    sigs = []
    for s in seqs:
        shapes = tuple(sorted(set(t[0] for t in s.split())))
        colors = tuple(sorted(set(t[1] for t in s.split())))
        sigs.append((shapes, colors))
    return sigs


# ------------------------- model def ----------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, l):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.out(h.squeeze(0))


# ---------------------- training utilities ----------------------------
def evaluate(model, loader, criterion, batch_size):
    model.eval()
    correct = total = loss_sum = 0
    with torch.no_grad():
        for b in loader:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            logits = model(b["input_ids"], b["lengths"])
            loss = criterion(logits, b["labels"])
            preds = logits.argmax(-1)
            correct += (preds == b["labels"]).sum().item()
            total += b["labels"].size(0)
            loss_sum += loss.item() * b["labels"].size(0)
    acc = correct / total
    return acc, loss_sum / total


def full_predictions(model, seqs, batch_size=64):
    preds = []
    model.eval()
    for i in range(0, len(seqs), batch_size):
        sub = seqs[i : i + batch_size]
        encs = [enc(s) for s in sub]
        lens = torch.tensor([len(x) for x in encs])
        mlen = lens.max()
        inp = torch.full((len(encs), mlen), vocab[PAD], dtype=torch.long)
        for j, row in enumerate(encs):
            inp[j, : len(row)] = torch.tensor(row)
        with torch.no_grad():
            logits = model(inp.to(device), lens.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


# ------------------- hyper-parameter sweep ----------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
epochs = 6
batch_size = 64
for wd in weight_decays:
    tag = str(wd)
    print(f"\n======== Training with weight_decay={wd} ========")
    data_entry = {
        "metrics": {"train": [], "dev": [], "test": {}, "NRGS": 0.0},
        "losses": {"train": [], "dev": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    loaders = {
        s: DataLoader(
            datasets[s],
            batch_size=batch_size,
            shuffle=(s == "train"),
            collate_fn=collate,
        )
        for s in ["train", "dev", "test"]
    }
    model = GRUClassifier(
        vocab_size, num_classes=len(set(raw_data["train"]["label"]))
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for b in loaders["train"]:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            logits = model(b["input_ids"], b["lengths"])
            loss = criterion(logits, b["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * b["labels"].size(0)
        avg_train = run_loss / len(datasets["train"])
        data_entry["losses"]["train"].append(avg_train)

        train_acc, _ = evaluate(model, loaders["train"], criterion, batch_size)
        dev_acc, dev_loss = evaluate(model, loaders["dev"], criterion, batch_size)
        data_entry["metrics"]["train"].append(train_acc)
        data_entry["metrics"]["dev"].append(dev_acc)
        data_entry["losses"]["dev"].append(dev_loss)
        data_entry["timestamps"].append(str(datetime.datetime.now()))
        print(
            f"Epoch {ep}: train_loss={avg_train:.4f}  dev_loss={dev_loss:.4f}  dev_acc={dev_acc:.3f}"
        )

    # ---------------- final test & NRGS -----------------
    test_acc, _ = evaluate(model, loaders["test"], criterion, batch_size)
    seqs = raw_data["test"]["sequence"]
    gts = raw_data["test"]["label"]
    preds = full_predictions(model, seqs, batch_size)
    swa = shape_weighted(seqs, gts, preds)
    cwa = color_weighted(seqs, gts, preds)
    train_sigs = set(signatures(raw_data["train"]["sequence"]))
    novel = [i for i, sig in enumerate(signatures(seqs)) if sig not in train_sigs]
    NRGS = sum(1 for i in novel if preds[i] == gts[i]) / len(novel) if novel else 0.0
    print(f"TEST  acc={test_acc:.3f}  SWA={swa:.3f}  CWA={cwa:.3f}  NRGS={NRGS:.3f}")

    data_entry["metrics"]["test"] = {"acc": test_acc, "swa": swa, "cwa": cwa}
    data_entry["metrics"]["NRGS"] = NRGS
    data_entry["predictions"] = preds
    data_entry["ground_truth"] = gts
    experiment_data["weight_decay"][tag] = data_entry
    torch.cuda.empty_cache()

# ---------------------- persist & plots -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as f:
    json.dump(experiment_data, f, indent=2)

# plot loss curves for each wd
for wd, dat in experiment_data["weight_decay"].items():
    plt.figure()
    plt.plot(dat["losses"]["train"], label="train")
    plt.plot(dat["losses"]["dev"], label="dev")
    plt.title(f"Loss curves (weight_decay={wd})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_wd_{wd}.png"))
    plt.close()
