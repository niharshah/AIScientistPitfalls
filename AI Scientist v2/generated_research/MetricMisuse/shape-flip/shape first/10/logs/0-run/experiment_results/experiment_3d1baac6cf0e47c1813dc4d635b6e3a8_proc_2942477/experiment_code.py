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

import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def CRWA(seqs, y_true, y_pred):
    weights = [count_shape(s) * count_color(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / max(sum(weights), 1)


def SWA(seqs, y_true, y_pred):
    weights = [count_shape(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / max(sum(weights), 1)


def CWA(seqs, y_true, y_pred):
    weights = [count_color(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / max(sum(weights), 1)


# ---------- dataset loading ----------
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


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH dataset.")
    except Exception as e:
        print("Could not load real data, generating synthetic toy data.", e)

        # synthetic
        def gen(n):
            shapes = "ABCD"
            colors = "abcd"
            data = []
            for i in range(n):
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(3, 10))
                )
                label = random.choice(["yes", "no"])
                data.append({"id": i, "sequence": seq, "label": label})
            return load_dataset("json", data_files={"train": data}, split="train")

        dset = DatasetDict()
        dset["train"] = gen(1000)
        dset["dev"] = gen(200)
        dset["test"] = gen(200)
    return dset


spr = get_dataset()

# ---------- vocab + label mapping ----------
all_tokens = set()
all_labels = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 reserved for PAD
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
        }  # raw for metric later


def collate(batch):
    max_len = max(x["length"] for x in batch).item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b["length"]
        padded[i, :l] = b["input_ids"]
        mask[i, :l] = 1
    out = {
        "input_ids": padded,
        "mask": mask,
        "n_shape": torch.stack([b["n_shape"] for b in batch]),
        "n_color": torch.stack([b["n_color"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }
    return out


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
        emb = self.emb(ids)  # B,L,E
        mask = mask.unsqueeze(-1)  # B,L,1
        summed = (emb * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        avg = summed / lengths
        x = torch.cat([avg, feat], dim=-1)
        return self.fc(x)


embed_dim = 64
model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)

# ---------- optimizer & loss ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


def evaluate(loader):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    loss_total, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
            logits = model(batch_t["input_ids"], batch_t["mask"], feat)
            loss = criterion(logits, batch_t["label"])
            loss_total += loss.item() * batch_t["label"].size(0)
            n += batch_t["label"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            labels = batch_t["label"].cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(labels)
            all_seq.extend(batch["raw_seq"])
    crwa = CRWA(all_seq, all_true, all_pred)
    swa = SWA(all_seq, all_true, all_pred)
    cwa = CWA(all_seq, all_true, all_pred)
    return loss_total / n, crwa, swa, cwa, all_true, all_pred, all_seq


# ---------- training loop ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    ep_loss, m = 0, 0
    for batch in train_loader:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
        optimizer.zero_grad()
        logits = model(batch_t["input_ids"], batch_t["mask"], feat)
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * batch_t["label"].size(0)
        m += batch_t["label"].size(0)
    train_loss = ep_loss / m

    val_loss, val_crwa, val_swa, val_cwa, y_true, y_pred, seqs = evaluate(dev_loader)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={val_crwa:.4f} | SWA={val_swa:.4f} | CWA={val_cwa:.4f}"
    )

    # store
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"CRWA": val_crwa, "SWA": val_swa, "CWA": val_cwa}
    )
    experiment_data["SPR_BENCH"]["predictions"].append(y_pred)
    experiment_data["SPR_BENCH"]["ground_truth"].append(y_true)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

# ---------- final test evaluation ----------
test_loss, test_crwa, test_swa, test_cwa, y_true, y_pred, seqs = evaluate(test_loader)
print(
    f"TEST: loss={test_loss:.4f} | CRWA={test_crwa:.4f} | SWA={test_swa:.4f} | CWA={test_cwa:.4f}"
)

experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "CRWA": test_crwa,
    "SWA": test_swa,
    "CWA": test_cwa,
}
experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
