# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, time, numpy as np, torch, copy, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def CRWA(seqs, y_true, y_pred):
    w = [count_shape(s) * count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def SWA(seqs, y_true, y_pred):
    w = [count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def CWA(seqs, y_true, y_pred):
    w = [count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
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

        dset = DatasetDict()
        dset["train"], dset["dev"], dset["test"] = gen(1000), gen(200), gen(200)
    return dset


spr = get_dataset()

# ---------- vocab ----------
all_tokens = set()
all_labels = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
label2id = {l: i for i, l in enumerate(sorted(all_labels))}
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
        ids = [tok2id[t] for t in self.seq[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "length": torch.tensor(len(ids)),
            "n_shape": torch.tensor(count_shape(self.seq[idx])),
            "n_color": torch.tensor(count_color(self.seq[idx])),
            "label": torch.tensor(self.lbl[idx]),
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
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(torch.cat([avg, feat], -1))


# ---------- helpers ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_s, all_t, all_p = [], [], []
    loss_total = n = 0
    with torch.no_grad():
        for batch in loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([b["n_shape"], b["n_color"]], -1).float()
            logits = model(b["input_ids"], b["mask"], feat)
            loss = criterion(logits, b["label"])
            loss_total += loss.item() * b["label"].size(0)
            n += b["label"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            labels = b["label"].cpu().numpy()
            all_p.extend(preds)
            all_t.extend(labels)
            all_s.extend(batch["raw_seq"])
    return (
        loss_total / n,
        CRWA(all_s, all_t, all_p),
        SWA(all_s, all_t, all_p),
        CWA(all_s, all_t, all_p),
        all_t,
        all_p,
        all_s,
    )


# ---------- hyperparameter tuning ----------
epoch_options = [5, 10, 20, 30]
experiment_data = {"num_epochs": {}}

for max_epochs in epoch_options:
    print(f"\n=== Training for up to {max_epochs} epochs ===")
    model = AvgEmbedClassifier(vocab_size, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    run_data = {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    best_val = float("inf")
    patience, pat_count = 3, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = m = 0
        for batch in train_loader:
            b = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([b["n_shape"], b["n_color"]], -1).float()
            optimizer.zero_grad()
            logits = model(b["input_ids"], b["mask"], feat)
            loss = criterion(logits, b["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * b["label"].size(0)
            m += b["label"].size(0)
        train_loss = ep_loss / m
        val_loss, val_crwa, val_swa, val_cwa, y_true, y_pred, seqs = evaluate(
            model, dev_loader
        )
        print(
            f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} CRWA={val_crwa:.4f}"
        )
        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["train"].append(None)
        run_data["metrics"]["val"].append(
            {"CRWA": val_crwa, "SWA": val_swa, "CWA": val_cwa}
        )
        run_data["predictions"].append(y_pred)
        run_data["ground_truth"].append(y_true)
        run_data["timestamps"].append(time.time())
        # early stopping
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            pat_count = 0
        else:
            pat_count += 1
        if pat_count >= patience:
            print("Early stopping.")
            break
    model.load_state_dict(best_state)
    test_loss, test_crwa, test_swa, test_cwa, yt, yp, seq = evaluate(model, test_loader)
    run_data["losses"]["test"] = test_loss
    run_data["metrics"]["test"] = {"CRWA": test_crwa, "SWA": test_swa, "CWA": test_cwa}
    print(
        f"TEST: loss={test_loss:.4f} CRWA={test_crwa:.4f} SWA={test_swa:.4f} CWA={test_cwa:.4f}"
    )
    experiment_data["num_epochs"][f"epochs_{max_epochs}"] = run_data

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
