import os, math, pathlib, random, json, time
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# -------------------- basic setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- load SPR_BENCH -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


# locate dataset folder
DEFAULT_PATHS = [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"),
]
for p in DEFAULT_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH folder not found")
print("Loading dataset from:", DATA_PATH)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# -------------------- metrics ------------------------
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


def weighted_acc(weights, y_true, y_pred):
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def cwa(seqs, y_true, y_pred):
    return weighted_acc([count_color_variety(s) for s in seqs], y_true, y_pred)


def swa(seqs, y_true, y_pred):
    return weighted_acc([count_shape_variety(s) for s in seqs], y_true, y_pred)


def ewa(seqs, y_true, y_pred):
    return weighted_acc([entropy_weight(s) for s in seqs], y_true, y_pred)


# -------------------- vocab / label ------------------
def build_vocab(seqs, min_freq=1):
    counter = Counter()
    for s in seqs:
        counter.update(s.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counter.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}
print("Vocab size:", len(vocab), " Num labels:", len(label2idx))


# -------------------- torch dataset -----------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, l2i):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]
        self.vocab = vocab
        self.l2i = l2i

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = [self.vocab.get(t, 1) for t in self.seq[idx].split()]
        return {
            "input_ids": torch.tensor(toks),
            "length": len(toks),
            "label": self.l2i[self.lab[idx]],
            "seq_raw": self.seq[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch)
    B = len(batch)
    pad = 0
    ids = torch.full((B, max_len), pad, dtype=torch.long)
    lengths, labels, raw = [], [], []
    for i, b in enumerate(batch):
        l = b["length"]
        ids[i, :l] = b["input_ids"]
        lengths.append(l)
        labels.append(b["label"])
        raw.append(b["seq_raw"])
    return {
        "input_ids": ids,
        "lengths": torch.tensor(lengths),
        "labels": torch.tensor(labels),
        "seq_raw": raw,
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


# -------------------- model --------------------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids) * (ids != 0).unsqueeze(-1)
        mean = x.sum(1) / lengths.unsqueeze(1).clamp(min=1).type_as(x)
        return self.fc(mean)


# -------------------- hyperparameter sweep -----------
learning_rates = [5e-4, 1e-3, 2e-3, 5e-3]
EPOCHS = 5
experiment_data = {"learning_rate": {}}

criterion = nn.CrossEntropyLoss()

for lr in learning_rates:
    print(f"\n======== LR = {lr:.0e} ========")
    lr_key = str(lr)
    experiment_data["learning_rate"][lr_key] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    model = MeanEmbedClassifier(len(vocab), 64, len(label2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- training epochs -----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0
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
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        train_loss = tot_loss / n
        experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["losses"]["train"].append(
            (epoch, train_loss)
        )
        # ----- validation -----
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
                labels = labs.cpu().tolist()
                all_seq.extend(batch["seq_raw"])
                all_true.extend([idx2label[i] for i in labels])
                all_pred.extend([idx2label[i] for i in preds])
        val_loss /= n
        experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["losses"]["val"].append(
            (epoch, val_loss)
        )
        cwa_score = cwa(all_seq, all_true, all_pred)
        swa_score = swa(all_seq, all_true, all_pred)
        ewa_score = ewa(all_seq, all_true, all_pred)
        experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["metrics"]["val"].append(
            (epoch, {"CWA": cwa_score, "SWA": swa_score, "EWA": ewa_score})
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"CWA={cwa_score:.4f} SWA={swa_score:.4f} EWA={ewa_score:.4f}"
        )

    # ----- test evaluation -----
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(ids, lens)
            preds = logits.argmax(1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_seq.extend(batch["seq_raw"])
            all_true.extend([idx2label[i] for i in labels])
            all_pred.extend([idx2label[i] for i in preds])
    experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["predictions"] = all_pred
    experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["ground_truth"] = all_true
    test_cwa, test_swa, test_ewa = (
        cwa(all_seq, all_true, all_pred),
        swa(all_seq, all_true, all_pred),
        ewa(all_seq, all_true, all_pred),
    )
    print(f"Test: CWA={test_cwa:.4f} SWA={test_swa:.4f} EWA={test_ewa:.4f}")
    # ----- plot losses for this lr -----
    ep = [
        e
        for e, _ in experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["losses"][
            "train"
        ]
    ]
    tr = [
        l
        for _, l in experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["losses"][
            "train"
        ]
    ]
    va = [
        l
        for _, l in experiment_data["learning_rate"][lr_key]["SPR_BENCH"]["losses"][
            "val"
        ]
    ]
    plt.figure()
    plt.plot(ep, tr, label="train")
    plt.plot(ep, va, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (lr={lr})")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, f"loss_curve_lr_{lr_key.replace('.','p')}.png")
    )
    plt.close()
    del model
    torch.cuda.empty_cache()

# -------------------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished. Data saved to working/experiment_data.npy")
