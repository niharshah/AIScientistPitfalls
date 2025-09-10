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


def _weighted_acc(seqs, y_true, y_pred, w_fn):
    w = [w_fn(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def CRWA(seqs, y_true, y_pred):
    return _weighted_acc(
        seqs, y_true, y_pred, lambda s: count_shape(s) * count_color(s)
    )


def SWA(seqs, y_true, y_pred):
    return _weighted_acc(seqs, y_true, y_pred, count_shape)


def CWA(seqs, y_true, y_pred):
    return _weighted_acc(seqs, y_true, y_pred, count_color)


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {split: _load(f"{split}.csv") for split in ["train", "dev", "test"]}
    )


def get_dataset():
    try:
        dset = load_spr_bench(
            pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        )
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

        dset = DatasetDict(train=gen(1000), dev=gen(200), test=gen(200))
    return dset


spr = get_dataset()

# ---------- vocab + label mapping ----------
all_tokens, all_labels = set(), set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 pad
label2id = {lab: i for i, lab in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size, num_classes = len(tok2id) + 1, len(label2id)
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


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    loss_tot, n = 0, 0
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        feat = torch.stack([b["n_shape"], b["n_color"]], -1).float()
        logits = model(b["input_ids"], b["mask"], feat)
        loss = criterion(logits, b["label"])
        loss_tot += loss.item() * b["label"].size(0)
        n += b["label"].size(0)
        preds = logits.argmax(-1).cpu().numpy()
        labels = b["label"].cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(labels)
        all_seq.extend(batch["raw_seq"])
    return (
        loss_tot / n if n else 0,
        CRWA(all_seq, all_true, all_pred),
        SWA(all_seq, all_true, all_pred),
        CWA(all_seq, all_true, all_pred),
        all_true,
        all_pred,
    )


# ---------- hyperparameter sweep ----------
smoothing_vals = [0.0, 0.05, 0.1, 0.2]
epochs = 5
embed_dim = 64
lr = 1e-3
experiment_data = {"label_smoothing": {}}

for sm in smoothing_vals:
    print(f"\n=== Training with label_smoothing={sm} ===")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=sm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0
        m = 0
        for batch in train_loader:
            b = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
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
        val_loss, crwa, swa, cwa, y_true, y_pred = evaluate(
            model, dev_loader, criterion
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={crwa:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
        )

        rec["losses"]["train"].append(train_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["train"].append(None)
        rec["metrics"]["val"].append({"CRWA": crwa, "SWA": swa, "CWA": cwa})
        rec["predictions"].append(y_pred)
        rec["ground_truth"].append(y_true)
        rec["timestamps"].append(time.time())

    test_loss, crwa, swa, cwa, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(
        f"TEST: loss={test_loss:.4f} | CRWA={crwa:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
    )
    rec["metrics"]["test"] = {"CRWA": crwa, "SWA": swa, "CWA": cwa}
    rec["losses"]["test"] = test_loss
    rec["predictions"].append(y_pred)
    rec["ground_truth"].append(y_true)
    experiment_data["label_smoothing"][f"{sm}"] = rec

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved results to", os.path.join(working_dir, "experiment_data.npy"))
