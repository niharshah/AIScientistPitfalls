import os, math, json, random, pathlib, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# ---------------------------- utils / reproducibility ----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------------------- load SPR_BENCH -------------------------------
def load_spr_bench(root: pathlib.Path):
    def _ld(name):  # helper
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = {}
    for split in ["train", "dev", "test"]:
        d[split] = _ld(f"{split}.csv")
    return d


DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for _p in DEFAULT_PATHS:
    if _p.exists():
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")

spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------------------- metrics --------------------------------------
def count_color_variety(seq):  # colour = 2nd char
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq):  # shape = 1st char
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def entropy_weight(seq):
    toks = seq.strip().split()
    if not toks:
        return 0.0
    freq = Counter(toks)
    tot = len(toks)
    return -sum((c / tot) * math.log2(c / tot) for c in freq.values())


def cwa(seq, y, yhat):
    w = [count_color_variety(s) for s in seq]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y, yhat)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def swa(seq, y, yhat):
    w = [count_shape_variety(s) for s in seq]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y, yhat)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def ewa(seq, y, yhat):
    w = [entropy_weight(s) for s in seq]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y, yhat)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ---------------------------- vocab / label --------------------------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    for s in seqs:
        cnt.update(s.strip().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab.setdefault(tok, len(vocab))
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Vocab size={len(vocab)} | num labels={len(label2idx)}")


# ---------------------------- dataset / loader -----------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, l2i):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.vocab = vocab
        self.l2i = l2i

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [self.vocab.get(tok, 1) for tok in self.seqs[idx].strip().split()]
        return {
            "input_ids": torch.tensor(toks),
            "length": torch.tensor(len(toks)),
            "label": torch.tensor(self.l2i[self.labels[idx]]),
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = 0
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lens, labels, seqs = [], [], []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        ids[i, :l] = b["input_ids"]
        lens.append(l)
        labels.append(b["label"])
        seqs.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lens),
        "labels": torch.stack(labels),
        "seq_raw": seqs,
    }


train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------------------------- model ----------------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vs, ed, nlab):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.fc = nn.Linear(ed, nlab)

    def forward(self, ids, lens):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        x = x * mask
        summed = x.sum(1)
        lens = lens.unsqueeze(1).type_as(summed)
        mean = summed / lens.clamp(min=1)
        return self.fc(mean)


# ---------------------------- training routine -----------------------------
def run_experiment(weight_decay_val, epochs=5, embed_dim=64, lr=1e-3):
    model = MeanEmbedClassifier(len(vocab), embed_dim, len(label2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_val)
    crit = nn.CrossEntropyLoss()
    exp_d = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        # train
        model.train()
        totloss = 0
        n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            lbl = batch["labels"].to(device)
            opt.zero_grad()
            logits = model(ids, lens)
            loss = crit(logits, lbl)
            loss.backward()
            opt.step()
            totloss += loss.item() * ids.size(0)
            n += ids.size(0)
        tr_loss = totloss / n
        exp_d["losses"]["train"].append((ep, tr_loss))

        # validate
        model.eval()
        vloss = 0
        n = 0
        seqs, tru, pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                lbl = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = crit(logits, lbl)
                vloss += loss.item() * ids.size(0)
                n += ids.size(0)
                p = logits.argmax(1).cpu().tolist()
                t = lbl.cpu().tolist()
                seqs.extend(batch["seq_raw"])
                tru.extend([idx2label[i] for i in t])
                pred.extend([idx2label[i] for i in p])
        v_loss = vloss / n
        exp_d["losses"]["val"].append((ep, v_loss))
        exp_d["metrics"]["val"].append(
            (
                ep,
                {
                    "CWA": cwa(seqs, tru, pred),
                    "SWA": swa(seqs, tru, pred),
                    "EWA": ewa(seqs, tru, pred),
                },
            )
        )
        print(
            f"[wd={weight_decay_val}] Ep{ep}: tr_loss={tr_loss:.4f} | "
            f"val_loss={v_loss:.4f} | CWA={exp_d['metrics']['val'][-1][1]['CWA']:.4f}"
        )

    # test evaluation
    model.eval()
    seqs, tru, pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            p = logits.argmax(1).cpu().tolist()
            t = batch["labels"].cpu().tolist()
            seqs.extend(batch["seq_raw"])
            tru.extend([idx2label[i] for i in t])
            pred.extend([idx2label[i] for i in p])
    exp_d["predictions"] = pred
    exp_d["ground_truth"] = tru
    exp_d["test_metrics"] = {
        "CWA": cwa(seqs, tru, pred),
        "SWA": swa(seqs, tru, pred),
        "EWA": ewa(seqs, tru, pred),
    }
    print(
        f"[wd={weight_decay_val}] Test CWA={exp_d['test_metrics']['CWA']:.4f}, "
        f"SWA={exp_d['test_metrics']['SWA']:.4f}, "
        f"EWA={exp_d['test_metrics']['EWA']:.4f}"
    )
    return exp_d


# ---------------------------- hyperparameter tuning ------------------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
experiment_data = {"weight_decay": {"SPR_BENCH": {}}}

for wd in weight_decays:
    run_data = run_experiment(wd)
    experiment_data["weight_decay"]["SPR_BENCH"][str(wd)] = run_data

# ---------------------------- save results ---------------------------------
np.save("experiment_data.npy", experiment_data)

# optional: plot a loss curve for each weight decay
colors = ["r", "g", "b", "m"]
plt.figure()
for col, wd in zip(colors, weight_decays):
    d = experiment_data["weight_decay"]["SPR_BENCH"][str(wd)]
    eps = [e for e, _ in d["losses"]["train"]]
    tr = [l for _, l in d["losses"]["train"]]
    val = [l for _, l in d["losses"]["val"]]
    plt.plot(eps, tr, color=col, linestyle="-", label=f"train wd={wd}")
    plt.plot(eps, val, color=col, linestyle="--", label=f"val wd={wd}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves (weight_decay)")
plt.legend(fontsize=8)
plt.savefig(os.path.join(working_dir, "weight_decay_loss_curves.png"))
plt.close()
