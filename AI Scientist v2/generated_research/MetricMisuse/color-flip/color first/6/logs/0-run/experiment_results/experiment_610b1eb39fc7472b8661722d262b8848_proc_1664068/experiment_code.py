import os, math, pathlib, random, time, json
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------- misc / seed --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------------------- data helpers -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = None
for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]:
    if p.exists():
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError("SPR_BENCH not found")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------------- metrics ------------------------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def entropy_weight(seq):
    toks = seq.split()
    total = len(toks)
    if not toks:
        return 0.0
    freqs = Counter(toks)
    return -sum((c / total) * math.log2(c / total) for c in freqs.values())


def _wa(weight_fn, seqs, y_true, y_pred):
    w = [weight_fn(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def cwa(s, y, p):
    return _wa(count_color_variety, s, y, p)


def swa(s, y, p):
    return _wa(count_shape_variety, s, y, p)


def ewa(s, y, p):
    return _wa(entropy_weight, s, y, p)


# ------------------- vocab / labels -----------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    [cnt.update(s.split()) for s in seqs]
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print("Vocab:", len(vocab), "Labels:", len(label2idx))


# ------------------- dataset ------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, ds, vocab, label2idx):
        self.seq = ds["sequence"]
        self.lab = ds["label"]
        self.vocab = vocab
        self.l2i = label2idx

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seq[idx].split()]
        return {
            "input_ids": torch.tensor(toks, dtype=torch.long),
            "length": torch.tensor(len(toks), dtype=torch.long),
            "label": torch.tensor(self.l2i[self.lab[idx]], dtype=torch.long),
            "seq_raw": self.seq[idx],
        }


def collate(batch):
    max_len = max(item["length"] for item in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lens, labels, raw = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lens.append(l)
        labels.append(b["label"])
        raw.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lens),
        "labels": torch.stack(labels),
        "seq_raw": raw,
    }


train_loader = lambda d: DataLoader(d, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = lambda d: DataLoader(d, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------- model --------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_labels)

    def forward(self, ids, lens):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        x = x * mask
        summed = x.sum(1)
        lens = lens.unsqueeze(1).type_as(summed)
        return self.fc(summed / lens.clamp(min=1))


# ------------------- experiment data ---------------
experiment_data = {"EPOCHS_tuning": {"SPR_BENCH": {}}}


def run_experiment(num_epochs):
    key = f"EPOCHS_{num_epochs}"
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    experiment_data["EPOCHS_tuning"]["SPR_BENCH"][key] = exp

    tr_ds, dev_ds, spr_test = [
        SPRTorchDataset(spr[s], vocab, label2idx) for s in ["train", "dev", "test"]
    ]
    tr_ld, dev_ld = train_loader(tr_ds), dev_loader(dev_ds)
    test_ld = dev_loader(spr_test)

    model = MeanEmbedClassifier(len(vocab), 64, len(label2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, num_epochs + 1):
        # train
        model.train()
        tot = 0
        n = 0
        for b in tr_ld:
            ids = b["input_ids"].to(device)
            lens = b["lengths"].to(device)
            labs = b["labels"].to(device)
            opt.zero_grad()
            logits = model(ids, lens)
            loss = crit(logits, labs)
            loss.backward()
            opt.step()
            tot += loss.item() * ids.size(0)
            n += ids.size(0)
        tr_loss = tot / n
        exp["losses"]["train"].append((ep, tr_loss))
        # val
        model.eval()
        tot = 0
        n = 0
        seqs, true, pred = [], [], []
        with torch.no_grad():
            for b in dev_ld:
                ids = b["input_ids"].to(device)
                lens = b["lengths"].to(device)
                labs = b["labels"].to(device)
                logits = model(ids, lens)
                loss = crit(logits, labs)
                tot += loss.item() * ids.size(0)
                n += ids.size(0)
                p = logits.argmax(1).cpu().tolist()
                t = labs.cpu().tolist()
                seqs.extend(b["seq_raw"])
                true.extend([idx2label[i] for i in t])
                pred.extend([idx2label[i] for i in p])
        val_loss = tot / n
        exp["losses"]["val"].append((ep, val_loss))
        exp["metrics"]["val"].append(
            (
                ep,
                {
                    "CWA": cwa(seqs, true, pred),
                    "SWA": swa(seqs, true, pred),
                    "EWA": ewa(seqs, true, pred),
                },
            )
        )
        print(
            f"[{key}] Epoch {ep}/{num_epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}"
        )
    # test
    model.eval()
    seqs, true, pred = [], [], []
    with torch.no_grad():
        for b in test_ld:
            ids = b["input_ids"].to(device)
            lens = b["lengths"].to(device)
            logits = model(ids, lens)
            p = logits.argmax(1).cpu().tolist()
            t = b["labels"].cpu().tolist()
            seqs.extend(b["seq_raw"])
            true.extend([idx2label[i] for i in t])
            pred.extend([idx2label[i] for i in p])
    exp["predictions"] = pred
    exp["ground_truth"] = true
    print(
        f"[{key}] Test  CWA={cwa(seqs,true,pred):.4f}  SWA={swa(seqs,true,pred):.4f}  EWA={ewa(seqs,true,pred):.4f}"
    )


# ------------------- run tuning --------------------
for ep_num in [5, 10, 15, 20, 25]:
    run_experiment(ep_num)

# ------------------- save & plot -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot: val loss curves for each run
plt.figure()
for key, data in experiment_data["EPOCHS_tuning"]["SPR_BENCH"].items():
    epochs = [e for e, _ in data["losses"]["val"]]
    vals = [l for _, l in data["losses"]["val"]]
    plt.plot(epochs, vals, label=key)
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Val Loss vs Epochs")
plt.legend()
plt.savefig(os.path.join(working_dir, "epochs_tuning_loss_curve.png"))
plt.close()
