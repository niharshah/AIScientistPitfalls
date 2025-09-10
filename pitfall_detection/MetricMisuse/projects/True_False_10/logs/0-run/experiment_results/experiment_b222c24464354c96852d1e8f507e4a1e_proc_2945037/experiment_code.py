import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- storage ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"embed_dim_tuning": {}}  # container required by instruction

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def CRWA(seqs, y_true, y_pred):
    w = [count_shape(s) * count_color(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def SWA(seqs, y_true, y_pred):
    w = [count_shape(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def CWA(seqs, y_true, y_pred):
    w = [count_color(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


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
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 pad
label2id = {lab: i for i, lab in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size = len(tok2id) + 1
num_classes = len(label2id)
print(f"Vocab size={vocab_size-1}, classes={num_classes}")


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lbl = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        ids = [tok2id[t] for t in toks]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(len(ids), dtype=torch.long),
            "n_shape": torch.tensor(count_shape(self.seq[idx]), dtype=torch.long),
            "n_color": torch.tensor(count_color(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    max_len = max(b["length"] for b in batch).item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["length"]
        padded[i, :l] = b["input_ids"]
        mask[i, :l] = 1
    return {
        "input_ids": padded,
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
    def __init__(self, vocab, embed_dim, num_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim + 2, num_cls)

    def forward(self, ids, mask, feat):
        emb = self.emb(ids)
        mask = mask.unsqueeze(-1)
        summed = (emb * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1)
        x = torch.cat([avg, feat], dim=-1)
        return self.fc(x)


# ---------- evaluation ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
            logits = model(batch_t["input_ids"], batch_t["mask"], feat)
            loss = criterion(logits, batch_t["label"])
            total_loss += loss.item() * batch_t["label"].size(0)
            n += batch_t["label"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            labels = batch_t["label"].cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(labels)
            all_seq.extend(batch["raw_seq"])
    return (
        total_loss / n,
        CRWA(all_seq, all_true, all_pred),
        SWA(all_seq, all_true, all_pred),
        CWA(all_seq, all_true, all_pred),
        all_true,
        all_pred,
        all_seq,
    )


# ---------- training procedure ----------
def run_training(embed_dim, epochs=5):
    print(f"\n--- Training with embed_dim={embed_dim} ---")
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        running_loss, n = 0, 0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
            optimizer.zero_grad()
            logits = model(batch_t["input_ids"], batch_t["mask"], feat)
            loss = criterion(logits, batch_t["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_t["label"].size(0)
            n += batch_t["label"].size(0)
        train_loss = running_loss / n
        val_loss, crwa, swa, cwa, y_true, y_pred, seqs = evaluate(model, dev_loader)
        print(
            f"Ep{ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={crwa:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
        )
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train"].append(None)
        exp["metrics"]["val"].append({"CRWA": crwa, "SWA": swa, "CWA": cwa})
        exp["predictions"].append(y_pred)
        exp["ground_truth"].append(y_true)
        exp["timestamps"].append(time.time())
    # final test
    test_loss, crwa, swa, cwa, y_true, y_pred, seqs = evaluate(model, test_loader)
    print(
        f"TEST (dim={embed_dim}): loss={test_loss:.4f} | CRWA={crwa:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
    )
    exp["metrics"]["test"] = {"CRWA": crwa, "SWA": swa, "CWA": cwa}
    exp["losses"]["test"] = test_loss
    exp["predictions_test"] = y_pred
    exp["ground_truth_test"] = y_true
    return exp, model


# ---------- hyperparameter sweep ----------
for dim in [32, 64, 128, 256]:
    exp_data, model = run_training(dim, epochs=5)
    experiment_data["embed_dim_tuning"][f"SPR_BENCH_dim_{dim}"] = exp_data
    # free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
