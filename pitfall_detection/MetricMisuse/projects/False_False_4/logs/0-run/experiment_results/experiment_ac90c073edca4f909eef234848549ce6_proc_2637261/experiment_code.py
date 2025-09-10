import os, pathlib, random, math, gc, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------
#  Basic setup
# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------------------------------
#  Dataset helpers (unchanged except for new rule-extraction functions)
# -------------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def seq_to_tokens(seq: str):
    return seq.strip().split()


def count_shape_variety(sequence: str) -> int:
    return len({tok[0] for tok in seq_to_tokens(sequence)})


def count_color_variety(sequence: str) -> int:
    return len({tok[1] for tok in seq_to_tokens(sequence) if len(tok) > 1})


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(1, sum(w))


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(1, sum(w))


def extract_shape_pattern(sequence: str):
    "Rule signature used for RGS – ordered shape symbols"
    return " ".join([tok[0] for tok in seq_to_tokens(sequence)])


# -------------------------------------------------------------------------
#  Load data
# -------------------------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print("Loaded sizes:", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------------
#  Vocabulary & label mapping
# -------------------------------------------------------------------------
vocab = {"<pad>": 0, "<unk>": 1}
for row in spr["train"]:
    for tok in seq_to_tokens(row["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)

label_set = sorted({row["label"] for row in spr["train"]})
label2id = {l: i for i, l in enumerate(label_set)}
num_classes = len(label_set)
print(f"Vocab size: {vocab_size} | num classes: {num_classes}")


# -------------------------------------------------------------------------
#  Dataset & DataLoader
# -------------------------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id):
        self.data = split
        self.vocab = vocab
        self.l2i = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = [
            self.vocab.get(tok, self.vocab["<unk>"])
            for tok in seq_to_tokens(row["sequence"])
        ]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.l2i[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    ids = [b["ids"] for b in batch]
    lengths = torch.tensor([len(x) for x in ids])
    ids_padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    raw_seq = [b["raw_seq"] for b in batch]
    return {"ids": ids_padded, "lengths": lengths, "label": labels, "raw_seq": raw_seq}


def build_loader(split, batch_size, shuffle):
    return DataLoader(
        SPRTorchDataset(split, vocab, label2id),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


# Hyper-parameters you may tune
BATCH_SIZE = 128
EMB_DIM = 128
LR = 3e-3
EPOCHS = 15
WEIGHT_DECAY = 1e-5

train_loader = build_loader(spr["train"], BATCH_SIZE, True)
dev_loader = build_loader(spr["dev"], BATCH_SIZE, False)
test_loader = build_loader(spr["test"], BATCH_SIZE, False)

# -------------------------------------------------------------------------
#  RGS mask FIX – build train rule set once, use shape-pattern criterion
# -------------------------------------------------------------------------
train_rule_set = {extract_shape_pattern(seq) for seq in spr["train"]["sequence"]}


def compute_rgs_mask(seqs):
    return np.array(
        [extract_shape_pattern(s) not in train_rule_set for s in seqs], dtype=bool
    )


# -------------------------------------------------------------------------
#  Model (unchanged architecture)
# -------------------------------------------------------------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)
        denom = mask.sum(1).clamp(min=1)
        avg = summed / denom
        return self.fc(avg)


# -------------------------------------------------------------------------
#  Metrics helper
# -------------------------------------------------------------------------
def compute_metrics(pred_np, true_np, seqs):
    acc = (pred_np == true_np).mean()
    swa = shape_weighted_accuracy(seqs, true_np, pred_np)
    cwa = color_weighted_accuracy(seqs, true_np, pred_np)
    wgma = math.sqrt(max(1e-8, swa * cwa))
    rgs_mask = compute_rgs_mask(seqs)
    rgs = (pred_np[rgs_mask] == true_np[rgs_mask]).mean() if rgs_mask.any() else 0.0
    return {"ACC": acc, "SWA": swa, "CWA": cwa, "WGMA": wgma, "RGS": rgs}


# -------------------------------------------------------------------------
#  Train / evaluate loops
# -------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, n_items = 0.0, 0
    all_pred, all_true, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            # move tensors to device
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["ids"])
            loss = criterion(logits, batch_t["label"])
            total_loss += loss.item() * batch_t["label"].size(0)
            n_items += batch_t["label"].size(0)
            preds = logits.argmax(-1)
            all_pred.extend(preds.cpu().numpy())
            all_true.extend(batch_t["label"].cpu().numpy())
            all_seqs.extend(batch_t["raw_seq"])
    metrics = compute_metrics(np.array(all_pred), np.array(all_true), all_seqs)
    metrics["loss"] = total_loss / n_items
    return metrics


# -------------------------------------------------------------------------
#  Experiment data container
# -------------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "dev": []},
        "losses": {"train": [], "dev": []},
        "predictions": {},
        "ground_truth": {},
    }
}

# -------------------------------------------------------------------------
#  Training
# -------------------------------------------------------------------------
model = AvgEmbedClassifier(vocab_size, EMB_DIM, num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_wgma, best_state = -1, None
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, running_items = 0.0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch_t["ids"])
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_t["label"].size(0)
        preds = logits.argmax(-1)
        running_correct += (preds == batch_t["label"]).sum().item()
        running_items += batch_t["label"].size(0)

    # epoch-level metrics
    train_loss = running_loss / running_items
    train_acc = running_correct / running_items
    train_metrics = {"loss": train_loss, "ACC": train_acc}

    dev_metrics = evaluate(model, dev_loader)

    # log / store
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["dev"].append(dev_metrics["loss"])
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_metrics)
    experiment_data["SPR_BENCH"]["metrics"]["dev"].append(dev_metrics)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"dev_loss={dev_metrics['loss']:.4f} | "
        f"ACC={dev_metrics['ACC']:.3f} | "
        f"WGMA={dev_metrics['WGMA']:.3f} | "
        f"RGS={dev_metrics['RGS']:.3f}"
    )

    # keep best model wrt WGMA
    if dev_metrics["WGMA"] > best_wgma:
        best_wgma = dev_metrics["WGMA"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# -------------------------------------------------------------------------
#  Evaluation on test split using best model
# -------------------------------------------------------------------------
model.load_state_dict(best_state)
test_metrics = evaluate(model, test_loader)
experiment_data["SPR_BENCH"]["metrics"]["test"] = test_metrics

print("\n==== FINAL TEST METRICS ====")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")


# Store predictions / labels for Dev & Test
def collect_preds(loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["ids"])
            preds.extend(logits.argmax(-1).cpu().numpy())
            truths.extend(batch_t["label"].cpu().numpy())
    return np.array(preds), np.array(truths)


(
    experiment_data["SPR_BENCH"]["predictions"]["dev"],
    experiment_data["SPR_BENCH"]["ground_truth"]["dev"],
) = collect_preds(dev_loader)
(
    experiment_data["SPR_BENCH"]["predictions"]["test"],
    experiment_data["SPR_BENCH"]["ground_truth"]["test"],
) = collect_preds(test_loader)

# -------------------------------------------------------------------------
#  Save artefacts
# -------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Plot loss curve
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["dev"], label="dev")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy")
plt.title("SPR_BENCH Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_spr.png"))
plt.close()

# Free GPU memory
del model, optimizer
torch.cuda.empty_cache()
gc.collect()
