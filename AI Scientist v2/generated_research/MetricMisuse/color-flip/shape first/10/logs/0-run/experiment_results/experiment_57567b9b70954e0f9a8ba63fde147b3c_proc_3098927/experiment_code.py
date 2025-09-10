import os, pathlib, random, string, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────── housekeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"dropout_rate": {}}  # will hold results per dropout value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ───────────────────────────── metrics helpers (copied)
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ───────────────────────────── dataset load (real or synthetic)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real dataset, generating synthetic data.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:5])
        colors = list("12345")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seq = " ".join(toks)
            maj_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            seqs.append(seq)
            labels.append(maj_shape)
        ids = list(range(n))
        return {"id": ids, "sequence": seqs, "label": labels}

    import datasets

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(2000)),
            "dev": datasets.Dataset.from_dict(synth_split(400)),
            "test": datasets.Dataset.from_dict(synth_split(400)),
        }
    )

# ───────────────────────────── vocab & encoding
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(dataset):
    global token2id, label2id
    toks, labs = set(), set()
    for s, l in zip(dataset["sequence"], dataset["label"]):
        toks.update(s.split())
        labs.add(l)
    token2id = {t: i + 1 for i, t in enumerate(sorted(toks))}
    label2id = {l: i for i, l in enumerate(sorted(labs))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):  # list[int]
    return [token2id[t] for t in seq.split()]


def encode_label(lab):  # int
    return label2id[lab]


# ───────────────────────────── torch dataset & dataloader
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "label_id": torch.tensor(encode_label(self.labels[idx]), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labs = torch.zeros(len(batch), dtype=torch.long)
    raws = []
    for i, itm in enumerate(batch):
        l = len(itm["input_ids"])
        ids[i, :l] = itm["input_ids"]
        labs[i] = itm["label_id"]
        raws.append(itm["raw_seq"])
    return {"input_ids": ids, "labels": labs, "raw_seq": raws}


# ───────────────────────────── model with dropout
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, num_labels, dim=64, dropout_rate=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(dim, num_labels)

    def forward(self, ids):
        emb = self.embed(ids)
        mask = (ids != PAD_ID).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ───────────────────────────── training utility
def run_experiment(drop_rate, epochs=5, batch_size=128):
    print(f"\n--- Training with dropout={drop_rate:.2f} ---")
    train_loader = DataLoader(
        SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
    )

    model = MeanPoolClassifier(
        len(token2id) + 1, len(label2id), dropout_rate=drop_rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = {"train_loss": [], "val_loss": [], "SWA": [], "CWA": [], "HWA": []}
    final_pred, final_true = [], []

    for ep in range(1, epochs + 1):
        # train
        model.train()
        tloss = 0.0
        for batch in train_loader:
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            optim.zero_grad()
            loss = criterion(model(ids), labels)
            loss.backward()
            optim.step()
            tloss += loss.item() * labels.size(0)
        tloss /= len(train_loader.dataset)
        metrics["train_loss"].append(tloss)

        # eval
        model.eval()
        vloss, y_t, y_p, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(ids)
                loss = criterion(logits, labels)
                vloss += loss.item() * labels.size(0)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                seqs.extend(batch["raw_seq"])
                y_t.extend([id2label[x] for x in labs])
                y_p.extend([id2label[x] for x in preds])
        vloss /= len(dev_loader.dataset)
        metrics["val_loss"].append(vloss)

        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        cwa = color_weighted_accuracy(seqs, y_t, y_p)
        hwa = 0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)
        metrics["SWA"].append(swa)
        metrics["CWA"].append(cwa)
        metrics["HWA"].append(hwa)

        print(
            f"Epoch {ep}: train_loss={tloss:.4f} val_loss={vloss:.4f} SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

        final_pred, final_true = y_p, y_t  # keep last epoch predictions

    return metrics, final_pred, final_true


# ───────────────────────────── hyperparameter sweep
dropout_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
best_hwa, best_rate = -1, None

for dr in dropout_grid:
    mtr, pred, true = run_experiment(dr)
    key = f"{dr:.2f}"
    experiment_data["dropout_rate"][key] = {
        "metrics": mtr,
        "predictions": pred,
        "ground_truth": true,
    }
    hwa_last = mtr["HWA"][-1]
    if hwa_last > best_hwa:
        best_hwa, best_rate = hwa_last, dr

print(f"\nBest dropout rate: {best_rate:.2f} with final HWA={best_hwa:.3f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
