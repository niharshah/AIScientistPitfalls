# dropout_rate hyperparameter-tuning baseline
import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -------------- reproducibility --------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------- work dir --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def _weighted_acc(seqs, y_true, y_pred, weight_fn):
    w = [weight_fn(s) for s in seqs]
    corr = [w0 if t == p else 0 for w0, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def CRWA(seqs, y_true, y_pred):
    return _weighted_acc(
        seqs, y_true, y_pred, lambda s: count_shape(s) * count_color(s)
    )


def SWA(seqs, y_true, y_pred):
    return _weighted_acc(seqs, y_true, y_pred, count_shape)


def CWA(seqs, y_true, y_pred):
    return _weighted_acc(seqs, y_true, y_pred, count_color)


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        d = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH dataset.")
    except Exception as e:
        print("Could not load real data, generating synthetic toy data.", e)

        def gen(n):
            shapes, colors = "ABCD", "abcd"
            data = [
                {
                    "id": i,
                    "sequence": " ".join(
                        random.choice(shapes) + random.choice(colors)
                        for _ in range(random.randint(3, 10))
                    ),
                    "label": random.choice(["yes", "no"]),
                }
                for i in range(n)
            ]
            return load_dataset("json", data_files={"train": data}, split="train")

        d = DatasetDict()
        d["train"], d["dev"], d["test"] = gen(1000), gen(200), gen(200)
    return d


spr = get_dataset()

# ---------- vocab & labels ----------
all_tokens = set()
all_labels = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
label2id = {lab: i for i, lab in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size, len_classes = len(tok2id) + 1, len(label2id)
print(f"Vocab size={vocab_size-1}, classes={len_classes}")


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lbl = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        ids = [tok2id[t] for t in toks]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(len(ids)),
            "n_shape": torch.tensor(count_shape(self.seq[idx])),
            "n_color": torch.tensor(count_color(self.seq[idx])),
            "label": torch.tensor(self.lbl[idx]),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    max_len = max(b["length"] for b in batch).item()
    pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["length"]
        pad[i, :l] = b["input_ids"]
        mask[i, :l] = 1
    return {
        "input_ids": pad,
        "mask": mask,
        "n_shape": torch.stack([b["n_shape"] for b in batch]),
        "n_color": torch.stack([b["n_color"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(spr["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab, embed_dim, num_cls, dropout_rate):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embed_dim + 2, num_cls)

    def forward(self, ids, mask, feat):
        emb = self.emb(ids)
        mask = mask.unsqueeze(-1)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        x = torch.cat([avg, feat], -1)
        x = self.drop(x)
        return self.fc(x)


# ---------- train / eval helpers ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_seq, all_t, all_p = [], [], []
    loss_tot = n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            feat = torch.stack([batch["n_shape"], batch["n_color"]], -1).float()
            logits = model(batch["input_ids"], batch["mask"], feat)
            loss = criterion(logits, batch["label"])
            loss_tot += loss.item() * batch["label"].size(0)
            n += batch["label"].size(0)
            pred = logits.argmax(-1).cpu().numpy()
            lbl = batch["label"].cpu().numpy()
            all_p.extend(pred)
            all_t.extend(lbl)
            all_seq.extend(batch["raw_seq"])
    return (
        loss_tot / n,
        CRWA(all_seq, all_t, all_p),
        SWA(all_seq, all_t, all_p),
        CWA(all_seq, all_t, all_p),
        all_t,
        all_p,
        all_seq,
    )


# ---------- experiment data dict ----------
experiment_data = {"dropout_rate": {}}

# ---------- hyperparameter grid ----------
embed_dim, epochs = 64, 5
drop_grid = [0.0, 0.1, 0.2, 0.3, 0.5]
best_val_crwa, best_state, best_rate = -1, None, None

for dp in drop_grid:
    print(f"\n=== Training with dropout={dp} ===")
    model = AvgEmbedClassifier(vocab_size, embed_dim, len_classes, dp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = m = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            feat = torch.stack([batch["n_shape"], batch["n_color"]], -1).float()
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["mask"], feat)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch["label"].size(0)
            m += batch["label"].size(0)
        train_loss = ep_loss / m
        val_loss, val_crwa, val_swa, val_cwa, y_t, y_p, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={val_crwa:.4f}"
        )
        hist["losses"]["train"].append(train_loss)
        hist["losses"]["val"].append(val_loss)
        hist["metrics"]["train"].append(None)
        hist["metrics"]["val"].append(
            {"CRWA": val_crwa, "SWA": val_swa, "CWA": val_cwa}
        )
        hist["predictions"].append(y_p)
        hist["ground_truth"].append(y_t)
        hist["timestamps"].append(time.time())
    experiment_data["dropout_rate"][dp] = {"SPR_BENCH": hist}
    if val_crwa > best_val_crwa:
        best_val_crwa, best_state, best_rate = val_crwa, model.state_dict(), dp

# ---------- test with best rate ----------
print(f"\nBest dropout rate by dev CRWA = {best_rate}")
best_model = AvgEmbedClassifier(vocab_size, embed_dim, len_classes, best_rate).to(
    device
)
best_model.load_state_dict(best_state)
test_loss, test_crwa, test_swa, test_cwa, y_t, y_p, _ = evaluate(
    best_model, test_loader
)
print(
    f"TEST: loss={test_loss:.4f} | CRWA={test_crwa:.4f} | SWA={test_swa:.4f} | CWA={test_cwa:.4f}"
)
experiment_data["dropout_rate"]["best_rate"] = best_rate
experiment_data["dropout_rate"]["test_metrics"] = {
    "loss": test_loss,
    "CRWA": test_crwa,
    "SWA": test_swa,
    "CWA": test_cwa,
}

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
