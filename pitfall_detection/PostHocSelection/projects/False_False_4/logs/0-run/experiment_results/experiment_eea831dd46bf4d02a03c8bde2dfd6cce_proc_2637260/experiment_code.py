import os, pathlib, random, time, math, json, gc
from datetime import datetime

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# ------------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------- mandatory working dir -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- mandatory device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =======================================================
#                SPR  HELPERS + DATA
# =======================================================
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    """Correctly return DatasetDict with three separate splits."""

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",  # treat each csv as one split
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def seq_to_tokens(seq: str):
    return seq.strip().split()


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")

spr = load_spr_bench(DATA_PATH)
print("Loaded split sizes:", {k: len(v) for k, v in spr.items()})

# ------------------- vocabulary ------------------------
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for tok in seq_to_tokens(ex["sequence"]):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)

labels = sorted({ex["label"] for ex in spr["train"]})
label2id = {l: i for i, l in enumerate(labels)}
num_classes = len(labels)
print(f"Vocab size = {vocab_size} | num_classes = {num_classes}")

# record tokens present in train split for RGS
train_tokens_set = {tok for ex in spr["train"]["sequence"] for tok in seq_to_tokens(ex)}


def compute_rgs_mask(seqs):
    """True where the sequence contains at least one token unseen in training."""
    return np.array(
        [any(tok not in train_tokens_set for tok in seq_to_tokens(s)) for s in seqs],
        dtype=bool,
    )


# ---------------- Torch Dataset -----------------------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, label2id):
        self.d = split
        self.v = vocab
        self.l2i = label2id

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        row = self.d[idx]
        ids = [self.v.get(t, self.v["<unk>"]) for t in seq_to_tokens(row["sequence"])]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.l2i[row["label"]], dtype=torch.long),
            "raw_seq": row["sequence"],
        }


def collate(batch):
    tensors = [b["ids"] for b in batch]
    lens = torch.tensor([len(t) for t in tensors], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    raws = [b["raw_seq"] for b in batch]
    return {"ids": padded, "lengths": lens, "label": labels, "raw_seq": raws}


batch_size = 256
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, label2id),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"], vocab, label2id),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2id),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


# ----------------- simple model -----------------------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, ids):
        emb = self.emb(ids)  # (B,L,D)
        mask = (ids != 0).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)
        avg = summed / mask.sum(1).clamp(min=1)
        return self.fc(avg)


# =======================================================
#                  TRAIN / EVAL
# =======================================================
criterion = nn.CrossEntropyLoss()


def run_eval(model, loader):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    preds_all, seqs_all, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            output = model(ids)
            loss = criterion(output, label)
            loss_sum += loss.item() * label.size(0)
            pred = output.argmax(-1)
            correct += (pred == label).sum().item()
            n += label.size(0)
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(label.cpu().tolist())
            seqs_all.extend(batch["raw_seq"])
    acc = correct / n
    return loss_sum / n, acc, np.array(preds_all), seqs_all, np.array(labels_all)


# ------------------ EXPERIMENT LOG ---------------------
experiment_data = {"SPR_BENCH": {"runs": {}}}

# ------------------ HYPER-PARAM SWEEP ------------------
embed_dims = [128]  # larger than before
lrs = [5e-4]  # slightly smaller
num_epochs = 25  # longer training

run_id = 0
for emb_dim in embed_dims:
    for lr in lrs:
        run_id += 1
        run_name = f"run{run_id}_emb{emb_dim}_lr{lr}"
        print(f"\n===== {run_name} =====")

        model = AvgEmbedClassifier(vocab_size, emb_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # per-run logging
        run_rec = {
            "train_loss": [],
            "dev_loss": [],
            "train_acc": [],
            "dev_acc": [],
            "swa": [],
            "cwa": [],
            "wgma": [],
        }

        for epoch in range(1, num_epochs + 1):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            for batch in train_loader:
                # move tensors
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                optimizer.zero_grad()
                logits = model(batch["ids"])
                loss = criterion(logits, batch["label"])
                loss.backward()
                optimizer.step()

                ep_loss += loss.item() * batch["label"].size(0)
                pred = logits.argmax(-1)
                correct += (pred == batch["label"]).sum().item()
                total += batch["label"].size(0)

            train_loss = ep_loss / total
            train_acc = correct / total

            dev_loss, dev_acc, dev_pred, dev_seqs, dev_true = run_eval(
                model, dev_loader
            )
            swa = shape_weighted_accuracy(dev_seqs, dev_true, dev_pred)
            cwa = color_weighted_accuracy(dev_seqs, dev_true, dev_pred)
            wgma = math.sqrt(max(swa, 1e-8) * max(cwa, 1e-8))  # avoid sqrt(0)

            run_rec["train_loss"].append(train_loss)
            run_rec["dev_loss"].append(dev_loss)
            run_rec["train_acc"].append(train_acc)
            run_rec["dev_acc"].append(dev_acc)
            run_rec["swa"].append(swa)
            run_rec["cwa"].append(cwa)
            run_rec["wgma"].append(wgma)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss {train_loss:.4f} dev_loss {dev_loss:.4f} "
                f"dev_acc {dev_acc:.3f} SWA {swa:.3f} CWA {cwa:.3f} WGMA {wgma:.3f}"
            )

        # ---- final test evaluation -----
        test_loss, test_acc, test_pred, test_seqs, test_true = run_eval(
            model, test_loader
        )
        test_swa = shape_weighted_accuracy(test_seqs, test_true, test_pred)
        test_cwa = color_weighted_accuracy(test_seqs, test_true, test_pred)
        test_wgma = math.sqrt(max(test_swa, 1e-8) * max(test_cwa, 1e-8))

        print(
            f"TEST | loss {test_loss:.4f} acc {test_acc:.3f} "
            f"SWA {test_swa:.3f} CWA {test_cwa:.3f} WGMA {test_wgma:.3f}"
        )

        run_rec["FINAL_TEST"] = {
            "loss": test_loss,
            "acc": test_acc,
            "swa": test_swa,
            "cwa": test_cwa,
            "wgma": test_wgma,
        }

        experiment_data["SPR_BENCH"]["runs"][run_name] = run_rec

        # ---- plotting -----
        plt.figure()
        plt.plot(run_rec["train_loss"], label="train")
        plt.plot(run_rec["dev_loss"], label="dev")
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropy")
        plt.title(f"Loss curve ({run_name})")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_{run_name}.png"))
        plt.close()

        # tidy
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

# ---------------- SAVE EVERYTHING ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
