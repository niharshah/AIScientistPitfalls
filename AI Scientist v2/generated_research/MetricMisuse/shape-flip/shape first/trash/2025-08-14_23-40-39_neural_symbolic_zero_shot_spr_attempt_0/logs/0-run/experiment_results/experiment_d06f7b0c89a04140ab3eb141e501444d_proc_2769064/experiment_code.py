import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# initial set-up and experiment bookkeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
# try to load real benchmark, else make synthetic fallback
def load_real_spr():
    try:
        import SPR  # the helper file provided in prompt is SPR.py

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            raise FileNotFoundError
        return SPR.load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Could not load real SPR_BENCH, reason:", e)
        return None


def make_synthetic_spr(n_train=800, n_dev=200, n_test=200):
    shapes = list("ABCDE")
    colors = list("XYZUV")

    def gen_seq():
        L = random.randint(4, 12)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def gen_split(n):
        seqs, labels = [], []
        for _ in range(n):
            seq = gen_seq()
            # simple synthetic rule: label 1 if more than 3 unique shapes else 0
            label = int(len(set(tok[0] for tok in seq.split())) > 3)
            seqs.append(seq)
            labels.append(label)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    from datasets import Dataset, DatasetDict

    d = DatasetDict()
    d["train"] = Dataset.from_dict(gen_split(n_train))
    d["dev"] = Dataset.from_dict(gen_split(n_dev))
    d["test"] = Dataset.from_dict(gen_split(n_test))
    return d


spr_bench = load_real_spr()
if spr_bench is None:
    spr_bench = make_synthetic_spr()
print("Loaded splits:", spr_bench.keys())


# ------------------------------------------------------------------
# helper metrics (copied from prompt)
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def balanced_poly_rule_score(swa, cwa):
    return math.sqrt(max(swa, 0) * max(cwa, 0))


# ------------------------------------------------------------------
# vocabulary build
all_tokens = set()
for split in ["train", "dev", "test"]:
    for seq in spr_bench[split]["sequence"]:
        all_tokens.update(seq.split())
token2id = {
    tok: i + 1 for i, tok in enumerate(sorted(all_tokens))
}  # 0 reserved for PAD
vocab_size = len(token2id) + 1
print("Vocabulary size:", vocab_size)

# label mapping (assume integers but ensure contiguous)
unique_labels = sorted(set(spr_bench["train"]["label"]))
label2id = {l: i for i, l in enumerate(unique_labels)}
num_classes = len(label2id)
print("Number of classes:", num_classes)


# ------------------------------------------------------------------
# dataset wrapper
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.max_len = max(len(s.split()) for s in self.seqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        ids = [token2id[t] for t in toks]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(b["input_ids"]) for b in batch]
    max_len = max(lengths)
    X = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        X[i, : lengths[i]] = b["input_ids"]
    y = torch.stack([b["label"] for b in batch])
    seqs = [b["seq"] for b in batch]
    return {"input_ids": X, "label": y, "seq": seqs}


train_ds = SPRDataset(spr_bench["train"])
dev_ds = SPRDataset(spr_bench["dev"])
BATCH = 64
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
dev_dl = DataLoader(dev_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------
# simple model: mean pooled embeddings -> linear
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, num_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        summed = (self.emb(x) * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        mean = summed / counts
        return self.fc(mean)


model = MeanPoolClassifier(vocab_size, 64, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------------------------------------
def evaluate(dataloader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input_ids"])
            loss = criterion(logits, batch_t["label"])
            total_loss += loss.item() * batch_t["label"].size(0)
            n += batch_t["label"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_t["label"].cpu().tolist())
            all_seqs.extend(batch["seq"])
    avg_loss = total_loss / n
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    bps = balanced_poly_rule_score(swa, cwa)
    return avg_loss, swa, cwa, bps, all_preds, all_labels, all_seqs


# ------------------------------------------------------------------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, seen = 0.0, 0
    for batch in train_dl:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["input_ids"])
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_t["label"].size(0)
        seen += batch_t["label"].size(0)
    train_loss = epoch_loss / seen
    val_loss, swa, cwa, bps, preds, labels, seqs = evaluate(dev_dl)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={swa:.3f} | CWA={cwa:.3f} | BPS={bps:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"SWA": swa, "CWA": cwa, "BPS": bps}
    )

# final predictions saved
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
