import os, math, time, json, random, pathlib
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------ working dir / device -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------ helper: load SPR_BENCH -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _load(f"{k}.csv") for k in ["train", "dev", "test"]})


DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in DEFAULT_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found.")
print("Loading dataset from:", DATA_PATH)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ------------ metrics --------------------------------
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def entropy_weight(seq):
    toks = seq.split()
    total = len(toks)
    if not total:
        return 0.0
    freqs = Counter(toks)
    return -sum((c / total) * math.log2(c / total) for c in freqs.values())


def weighted_acc(weight_func, seqs, y_true, y_pred):
    w = [weight_func(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def cwa(s, y, p):
    return weighted_acc(count_color_variety, s, y, p)


def swa(s, y, p):
    return weighted_acc(count_shape_variety, s, y, p)


def ewa(s, y, p):
    return weighted_acc(entropy_weight, s, y, p)


# ------------ vocab / label mapping ------------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    [cnt.update(s.split()) for s in seqs]
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
print("Vocab size:", len(vocab))
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
num_labels = len(label2idx)


# ------------ Torch Dataset --------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, l2i):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.l2i = vocab, l2i

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [self.vocab.get(t, 1) for t in self.seqs[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "length": torch.tensor(len(ids)),
            "label": torch.tensor(self.l2i[self.labels[idx]]),
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lengths, labels, seq_raw = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lengths.append(l)
        labels.append(b["label"])
        seq_raw.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lengths),
        "labels": torch.stack(labels),
        "seq_raw": seq_raw,
    }


train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, label2idx),
    batch_size=64,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"], vocab, label2idx),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2idx),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)


# ------------ Model ----------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, n_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, n_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)
        summed = (x * mask).sum(1)
        mean = summed / lengths.unsqueeze(1).type_as(summed).clamp(min=1)
        return self.fc(mean)


# ------------ training / evaluation ------------------
def run_experiment(embed_dim, epochs=5):
    model = MeanEmbedClassifier(len(vocab), embed_dim, num_labels).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = {"losses": {"train": [], "val": []}, "metrics": {"train": [], "val": []}}
    best_dev_ewa = -1
    best_state = None
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tot_loss = 0
        n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            opt.zero_grad()
            logits = model(ids, lens)
            loss = crit(logits, labs)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        rec["losses"]["train"].append((epoch, tot_loss / n))
        # val
        model.eval()
        v_loss = 0
        n = 0
        seqs = true = pred = []
        seqs = []
        true = []
        pred = []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                labs = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = crit(logits, labs)
                v_loss += loss.item() * ids.size(0)
                n += ids.size(0)
                ps = logits.argmax(1).cpu().tolist()
                lbs = labs.cpu().tolist()
                seqs.extend(batch["seq_raw"])
                true.extend([idx2label[i] for i in lbs])
                pred.extend([idx2label[i] for i in ps])
        v_loss /= n
        c, s, e = cwa(seqs, true, pred), swa(seqs, true, pred), ewa(seqs, true, pred)
        rec["losses"]["val"].append((epoch, v_loss))
        rec["metrics"]["val"].append((epoch, {"CWA": c, "SWA": s, "EWA": e}))
        print(
            f"[dim={embed_dim}] Epoch {epoch}: train_loss={tot_loss/n:.4f} "
            f"val_loss={v_loss:.4f} CWA={c:.4f} SWA={s:.4f} EWA={e:.4f}"
        )
        if e > best_dev_ewa:
            best_dev_ewa = e
            best_state = model.state_dict()
    # load best for test
    model.load_state_dict(best_state)
    model.eval()
    seqs = true = pred = []
    seqs = []
    true = []
    pred = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            ps = logits.argmax(1).cpu().tolist()
            lbs = batch["labels"].cpu().tolist()
            seqs.extend(batch["seq_raw"])
            true.extend([idx2label[i] for i in lbs])
            pred.extend([idx2label[i] for i in ps])
    test_c, test_s, test_e = (
        cwa(seqs, true, pred),
        swa(seqs, true, pred),
        ewa(seqs, true, pred),
    )
    rec["predictions"] = pred
    rec["ground_truth"] = true
    rec["test_metrics"] = {"CWA": test_c, "SWA": test_s, "EWA": test_e}
    print(f"[dim={embed_dim}] TEST  CWA={test_c:.4f} SWA={test_s:.4f} EWA={test_e:.4f}")
    del model
    torch.cuda.empty_cache()
    return rec, best_dev_ewa


# ------------ Hyperparameter sweep -------------------
embed_dims = [32, 64, 128, 256]
experiment_data = {"embedding_dim": {"SPR_BENCH": {}}}
best_dim = None
best_score = -1
for dim in embed_dims:
    record, dev_ewa = run_experiment(dim)
    experiment_data["embedding_dim"]["SPR_BENCH"][str(dim)] = record
    if dev_ewa > best_score:
        best_score, best_dim = dev_ewa, dim

print(f"Best embedding_dim on dev EWA: {best_dim} (EWA={best_score:.4f})")

# ------------ save results ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# optional plot of losses for best dim
best_rec = experiment_data["embedding_dim"]["SPR_BENCH"][str(best_dim)]
epochs = [e for e, _ in best_rec["losses"]["train"]]
tr = [l for _, l in best_rec["losses"]["train"]]
vl = [l for _, l in best_rec["losses"]["val"]]
plt.figure()
plt.plot(epochs, tr, label="train")
plt.plot(epochs, vl, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss Curve (emb_dim={best_dim})")
plt.legend()
plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
plt.close()
