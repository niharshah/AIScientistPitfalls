# ---------------------------------------------------------------
#  Hyper-parameter tuning: learning-rate sweep for SPR_BENCH task
# ---------------------------------------------------------------
import os, random, string, datetime, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------
#  Experiment bookkeeping
experiment_data = {"learning_rate_tuning": {"SPR_BENCH": {}}}

# ---------------------------------------------
#  Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------------------------
#  Data (load real SPR_BENCH or create synthetic)
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


use_synth = not spr_files_exist(SPR_PATH)

if use_synth:
    print("Real SPR_BENCH not found – using synthetic.")
    shapes = string.ascii_uppercase[:6]  # A-F
    colors = "0123"

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(seq):
        us = len(set(t[0] for t in seq.split()))
        uc = len(set(t[1] for t in seq.split()))
        return int(us == uc)

    def make_split(n):
        s = [rand_seq() for _ in range(n)]
        return {"sequence": s, "label": [rule(x) for x in s]}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    print("Loading real SPR_BENCH")
    from datasets import load_dataset, DatasetDict

    def load_csv(name):
        return load_dataset(
            "csv",
            data_files=os.path.join(SPR_PATH, name),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict(
        train=load_csv("train.csv"), dev=load_csv("dev.csv"), test=load_csv("test.csv")
    )
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }


# ---------------------------------------------
#  Metrics helpers
def count_shape(seq):
    return len(set(t[0] for t in seq.split()))


def count_color(seq):
    return len(set(t[1] for t in seq.split()))


def swa(seqs, y, p):
    w = [count_shape(s) for s in seqs]
    return sum(wi * (ti == pi) for wi, ti, pi in zip(w, y, p)) / max(sum(w), 1)


def cwa(seqs, y, p):
    w = [count_color(s) for s in seqs]
    return sum(wi * (ti == pi) for wi, ti, pi in zip(w, y, p)) / max(sum(w), 1)


def signatures(seqs):
    sig = []
    for s in seqs:
        sig.append(
            (
                tuple(sorted(set(t[0] for t in s.split()))),
                tuple(sorted(set(t[1] for t in s.split()))),
            )
        )
    return sig


# ---------------------------------------------
#  Vocab / tokenizer
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ---------------------------------------------
#  Dataset / Dataloader
class SPRSet(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"ids": self.X[i], "label": self.y[i]}


def collate(batch):
    lens = [len(b["ids"]) for b in batch]
    maxlen = max(lens)
    inp = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        inp[i, : lens[i]] = b["ids"]
        labels[i] = b["label"]
    return {"ids": inp, "lengths": torch.tensor(lens), "labels": labels}


datasets = {
    sp: SPRSet(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}


# ---------------------------------------------
#  Model
class GRUClassifier(nn.Module):
    def __init__(self, vsz, emb, hid, classes):
        super().__init__()
        self.emb = nn.Embedding(vsz, emb, padding_idx=vocab[PAD])
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, classes)

    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.out(h.squeeze(0))


# ---------------------------------------------
#  Training / evaluation routine
def run_training(lr, epochs=6, batch_size=64):
    model = GRUClassifier(len(vocab), 64, 128, len(set(raw_data["train"]["label"]))).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loaders = {
        sp: DataLoader(
            datasets[sp],
            batch_size=batch_size,
            shuffle=(sp == "train"),
            collate_fn=collate,
        )
        for sp in ["train", "dev", "test"]
    }
    losses = {"train": [], "val": []}
    metrics = {"train": [], "val": []}  # accuracy
    for ep in range(epochs):
        model.train()
        tr_loss = 0
        correct = tot = 0
        for batch in loaders["train"]:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"], batch["lengths"])
            loss = crit(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch["labels"].size(0)
            pred = logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            tot += batch["labels"].size(0)
        losses["train"].append(tr_loss / len(datasets["train"]))
        metrics["train"].append(correct / tot)

        # dev
        model.eval()
        dv_loss = 0
        correct = tot = 0
        with torch.no_grad():
            for batch in loaders["dev"]:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["ids"], batch["lengths"])
                loss = crit(logits, batch["labels"])
                dv_loss += loss.item() * batch["labels"].size(0)
                pred = logits.argmax(-1)
                correct += (pred == batch["labels"]).sum().item()
                tot += batch["labels"].size(0)
        losses["val"].append(dv_loss / len(datasets["dev"]))
        metrics["val"].append(correct / tot)
        print(
            f"lr={lr:.0e}  epoch={ep+1}  "
            f"train_loss={losses['train'][-1]:.4f}  "
            f"val_loss={losses['val'][-1]:.4f}  val_acc={metrics['val'][-1]:.3f}"
        )
    # final test
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for batch in loaders["test"]:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"], batch["lengths"])
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    seqs = raw_data["test"]["sequence"]
    test_acc = sum(p == t for p, t in zip(preds, gts)) / len(gts)
    test_swa = swa(seqs, gts, preds)
    test_cwa = cwa(seqs, gts, preds)
    # NRGS
    train_sig = set(signatures(raw_data["train"]["sequence"]))
    test_sig = signatures(seqs)
    novel = [i for i, s in enumerate(test_sig) if s not in train_sig]
    NRGS = sum(preds[i] == gts[i] for i in novel) / len(novel) if novel else 0.0
    return {
        "losses": losses,
        "metrics": {
            "train": metrics["train"],
            "val": metrics["val"],
            "test": {"acc": test_acc, "swa": test_swa, "cwa": test_cwa},
        },
        "predictions": preds,
        "ground_truth": gts,
        "NRGS": NRGS,
    }


# ---------------------------------------------
#  Learning-rate sweep
lr_values = [3e-4, 5e-4, 1e-3, 2e-3]
best_lr, best_val = float("inf"), None
for lr in lr_values:
    res = run_training(lr)
    experiment_data["learning_rate_tuning"]["SPR_BENCH"][f"lr_{lr:.0e}"] = res
    final_val_loss = res["losses"]["val"][-1]
    if final_val_loss < best_lr:
        best_lr, best_val = final_val_loss, lr
print(f"Best learning-rate: {best_val}  (val_loss={best_lr:.4f})")

# ---------------------------------------------
#  Save everything
np.save("experiment_data.npy", experiment_data, allow_pickle=True)
with open("experiment_data.json", "w") as fp:
    json.dump(experiment_data, fp, indent=2)

#  Plot losses for each lr
for lr in lr_values:
    lst = experiment_data["learning_rate_tuning"]["SPR_BENCH"][f"lr_{lr:.0e}"]["losses"]
    plt.plot(lst["train"], label=f"train lr={lr:.0e}")
    plt.plot(lst["val"], "--", label=f"val lr={lr:.0e}")
plt.legend()
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curves_sweep.png")
plt.close()
