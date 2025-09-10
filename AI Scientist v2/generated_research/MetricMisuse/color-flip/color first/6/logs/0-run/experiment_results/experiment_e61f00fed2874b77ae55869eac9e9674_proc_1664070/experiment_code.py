# Dropout-rate hyper-parameter tuning for SPR-BENCH
import os, math, json, random, pathlib, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# ---------------------- housekeeping / device -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- helper to load SPR_BENCH --------------------------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


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
print(f"Loading dataset from: {DATA_PATH}")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# --------------------- metrics --------------------------------------------
def count_color_variety(seq):  # colour variety weight
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):  # shape variety weight
    return len(set(tok[0] for tok in seq.split() if tok))


def entropy_weight(seq):
    toks = seq.split()
    total = len(toks)
    if total == 0:
        return 0.0
    freqs = Counter(toks)
    ent = -sum((c / total) * math.log2(c / total) for c in freqs.values())
    return ent


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


def ewa(seqs, y_t, y_p):
    w = [entropy_weight(s) for s in seqs]
    cor = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(cor) / sum(w) if sum(w) > 0 else 0.0


# --------------------- vocab / label map ----------------------------------
def build_vocab(seqs, min_freq=1):
    cnt = Counter()
    for s in seqs:
        cnt.update(s.split())
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
print("Num labels:", num_labels)


# --------------------- torch Dataset / DataLoader -------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, vocab, label2idx):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]
        self.vocab = vocab
        self.l2i = label2idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seqs[idx].split()]
        return {
            "input_ids": torch.tensor(toks),
            "length": torch.tensor(len(toks)),
            "label": torch.tensor(self.l2i[self.labels[idx]]),
            "seq_raw": self.seqs[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    pad = 0
    ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    lens, labs, raws = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lens.append(l)
        labs.append(b["label"])
        raws.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.stack(lens),
        "labels": torch.stack(labs),
        "seq_raw": raws,
    }


train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# --------------------- Model with Dropout ---------------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_lbls, dropout_rate=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(emb_dim, num_lbls)

    def forward(self, ids, lens):
        x = self.emb(ids)  # (B,T,D)
        mask = (ids != 0).unsqueeze(-1)
        x = x * mask
        summed = x.sum(1)
        lens = lens.unsqueeze(1).type_as(summed)
        mean = summed / lens.clamp(min=1)
        mean = self.dropout(mean)
        return self.fc(mean)


# --------------------- hyper-parameter grid -------------------------------
dropout_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS = 5
experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}

for p_drop in dropout_grid:
    print(f"\n=== Training with dropout p={p_drop} ===")
    model = MeanEmbedClassifier(len(vocab), 64, num_labels, dropout_rate=p_drop).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(ids, lens)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        train_loss = total_loss / n
        run_data["losses"]["train"].append((epoch, train_loss))

        # ---- dev ----
        model.eval()
        val_loss = 0
        n = 0
        all_seq, all_true, all_pred = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                lens = batch["lengths"].to(device)
                labs = batch["labels"].to(device)
                logits = model(ids, lens)
                loss = criterion(logits, labs)
                val_loss += loss.item() * ids.size(0)
                n += ids.size(0)
                preds = logits.argmax(1).cpu().tolist()
                labs_cpu = labs.cpu().tolist()
                all_seq.extend(batch["seq_raw"])
                all_true.extend([idx2label[i] for i in labs_cpu])
                all_pred.extend([idx2label[i] for i in preds])
        val_loss /= n
        run_data["losses"]["val"].append((epoch, val_loss))
        cwa_s = cwa(all_seq, all_true, all_pred)
        swa_s = swa(all_seq, all_true, all_pred)
        ewa_s = ewa(all_seq, all_true, all_pred)
        run_data["metrics"]["val"].append(
            (epoch, {"CWA": cwa_s, "SWA": swa_s, "EWA": ewa_s})
        )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"CWA={cwa_s:.4f}, SWA={swa_s:.4f}, EWA={ewa_s:.4f}"
        )

    # ------------ test evaluation -----------------------------------------
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            preds = logits.argmax(1).cpu().tolist()
            labs = batch["labels"].cpu().tolist()
            all_seq.extend(batch["seq_raw"])
            all_true.extend([idx2label[i] for i in labs])
            all_pred.extend([idx2label[i] for i in preds])
    run_data["predictions"] = all_pred
    run_data["ground_truth"] = all_true
    test_cwa = cwa(all_seq, all_true, all_pred)
    test_swa = swa(all_seq, all_true, all_pred)
    test_ewa = ewa(all_seq, all_true, all_pred)
    print(f"Test:  CWA={test_cwa:.4f}, SWA={test_swa:.4f}, EWA={test_ewa:.4f}")
    run_data["metrics"]["test"] = {"CWA": test_cwa, "SWA": test_swa, "EWA": test_ewa}

    # save loss curve for this dropout
    ep = [e for e, _ in run_data["losses"]["train"]]
    tr = [l for _, l in run_data["losses"]["train"]]
    vl = [l for _, l in run_data["losses"]["val"]]
    plt.figure()
    plt.plot(ep, tr, label="train")
    plt.plot(ep, vl, label="val")
    plt.title(f"Loss curve p={p_drop}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_p{p_drop}.png"))
    plt.close()

    # store run data
    experiment_data["dropout_rate"]["SPR_BENCH"][f"p={p_drop}"] = run_data

# --------------------- save experiment data --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to working/experiment_data.npy")
