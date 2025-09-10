import os, sys, random, math, time, json
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device handling =============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Try loading real SPR_BENCH; fall back to synthetic if unavailable
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = os.environ.get("SPR_DATA_PATH", "./SPR_BENCH")
    dataset = load_spr_bench(os.path.abspath(DATA_PATH))
    print("Loaded SPR_BENCH from", DATA_PATH)
except Exception as e:
    print("Could not load SPR_BENCH, generating synthetic data. Error:", e)

    # -------------------------------------------------------------------------
    # create minimal replacements for metric helpers
    def count_shape_variety(sequence):
        return len(set(tok[0] for tok in sequence.strip().split() if tok))

    def count_color_variety(sequence):
        return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_shape_variety(s) for s in seqs]
        c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) or 1.0)

    def color_weighted_accuracy(seqs, y_true, y_pred):
        w = [count_color_variety(s) for s in seqs]
        c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
        return sum(c) / (sum(w) or 1.0)

    # synthetic dataset generator ---------------------------------------------
    def synth_dataset(n):
        shapes = ["A", "B", "C", "D"]
        colors = ["X", "Y", "Z"]
        rows = []
        for i in range(n):
            length = random.randint(5, 15)
            seq = []
            for _ in range(length):
                seq.append(random.choice(shapes) + random.choice(colors))
            sequence = " ".join(seq)
            label = random.choice(["rule0", "rule1"])  # dummy 2-class
            rows.append({"id": str(i), "sequence": sequence, "label": label})
        return rows

    class ListDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    dataset = {
        "train": ListDataset(synth_dataset(2000)),
        "dev": ListDataset(synth_dataset(400)),
        "test": ListDataset(synth_dataset(400)),
    }
# -----------------------------------------------------------------------------

# Build vocabulary ------------------------------------------------------------
all_tokens = Counter()
for row in dataset["train"]:
    all_tokens.update(row["sequence"].split())

vocab = {"<pad>": 0, "<unk>": 1}
for tok in sorted(all_tokens):
    vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq):
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq.split()]


# label mapping
all_labels = sorted(
    {row["label"] for split in ["train", "dev", "test"] for row in dataset[split]}
)
label2id = {lbl: i for i, lbl in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)


# Datasets / Dataloaders -------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "input_ids": torch.tensor(encode(row["sequence"]), dtype=torch.long),
            "label": torch.tensor(label2id[row["label"]], dtype=torch.long),
            "sequence": row["sequence"],  # keep original for metrics
        }


def collate(batch):
    maxlen = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    mask = []
    labels = []
    seqs = []
    for item in batch:
        seq = item["input_ids"]
        pad_len = maxlen - len(seq)
        input_ids.append(torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)]))
        mask.append(
            torch.cat(
                [
                    torch.ones(len(seq), dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ]
            )
        )
        labels.append(item["label"])
        seqs.append(item["sequence"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(mask),
        "labels": torch.stack(labels),
        "sequences": seqs,
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(dataset["train"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(dataset["dev"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(dataset["test"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# Model -----------------------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        emb = self.embed(input_ids)  # B x T x D
        mask = attention_mask.unsqueeze(-1)  # B x T x 1
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        mean = summed / denom
        logits = self.classifier(mean)  # B x C
        return logits


model = MeanPoolClassifier(vocab_size, embed_dim=64, num_labels=num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Experiment data structure ---------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": defaultdict(list),
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# Training loop ---------------------------------------------------------------
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["input_ids"], batch_t["attention_mask"])
            loss = criterion(logits, batch_t["labels"])
            total_loss += loss.item() * logits.size(0)
            preds = logits.argmax(-1).cpu().tolist()
            labels = batch_t["labels"].cpu().tolist()
            seqs = batch["sequences"]
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(seqs)
    avg_loss = total_loss / len(loader.dataset)
    swa = shape_weighted_accuracy(
        all_seqs, [id2label[i] for i in all_labels], [id2label[i] for i in all_preds]
    )
    cwa = color_weighted_accuracy(
        all_seqs, [id2label[i] for i in all_labels], [id2label[i] for i in all_preds]
    )
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return avg_loss, swa, cwa, hwa, all_preds, all_labels, all_seqs


epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["input_ids"], batch_t["attention_mask"])
        loss = criterion(logits, batch_t["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * logits.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    val_loss, swa, cwa, hwa, _, _, _ = evaluate(dev_loader)

    # logging
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_cwa"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_hwa"].append(hwa)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
    )

# Final test evaluation -------------------------------------------------------
test_loss, swa, cwa, hwa, preds, gts, seqs = evaluate(test_loader)
print(f"Test: loss={test_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}")

experiment_data["SPR_BENCH"]["metrics"]["test_swa"] = swa
experiment_data["SPR_BENCH"]["metrics"]["test_cwa"] = cwa
experiment_data["SPR_BENCH"]["metrics"]["test_hwa"] = hwa
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# Save experiment data --------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
