import os, random, string, datetime, json, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------- house-keeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "num_epochs": {  # hyper-parameter tuned
        "SPR_BENCH": {
            "metrics": {"train": [], "val": [], "test": [], "NRGS": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# -------------------------- device -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- load/generate SPR_BENCH -------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{split}.csv"))
        for split in ["train", "dev", "test"]
    )


use_synthetic = not spr_files_exist(SPR_PATH)
if use_synthetic:
    print("Real SPR_BENCH not found – generating synthetic data.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = [str(i) for i in range(4)]  # 0-3

    def random_seq():
        length = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )

    def rule_label(seq):
        us = len(set(tok[0] for tok in seq.split()))
        uc = len(set(tok[1] for tok in seq.split()))
        return int(us == uc)

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        labels = [rule_label(s) for s in seqs]
        return {"sequence": seqs, "label": labels}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    print("Loading real SPR_BENCH")
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root: str):
        def _load(split_csv):
            return load_dataset(
                "csv", data_files=os.path.join(root, split_csv), split="train"
            )

        ds = DatasetDict()
        for sp in ["train", "dev", "test"]:
            ds[sp] = _load(f"{sp}.csv")
        return ds

    ds = load_spr_bench(SPR_PATH)
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }

# --------------------- helper metrics --------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def compute_signatures(seqs):
    sigs = []
    for s in seqs:
        shapes = tuple(sorted(set(tok[0] for tok in s.split())))
        colors = tuple(sorted(set(tok[1] for tok in s.split())))
        sigs.append((shapes, colors))
    return sigs


# ------------------------ vocab/tokenizer ----------------------------
def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


# ----------------------------- dataset -------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = [torch.tensor(encode_sequence(s), dtype=torch.long) for s in sequences]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw_seq = sequences

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "label": self.y[idx]}


def collate(batch):
    lengths = [len(item["input_ids"]) for item in batch]
    maxlen = max(lengths)
    input_ids = torch.full(
        (len(batch), maxlen), fill_value=vocab[PAD], dtype=torch.long
    )
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
        labels[i] = item["label"]
    return {"input_ids": input_ids, "labels": labels, "lengths": torch.tensor(lengths)}


datasets = {
    sp: SPRTorchDataset(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}

batch_size = 64
loaders = {
    sp: DataLoader(
        datasets[sp],
        batch_size=batch_size,
        shuffle=(sp == "train"),
        collate_fn=collate,
    )
    for sp in ["train", "dev", "test"]
}


# ------------------------ model --------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.out(h.squeeze(0))


num_classes = len(set(raw_data["train"]["label"]))
model = GRUClassifier(vocab_size, 64, 128, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------- evaluation helper ----------------------------
@torch.no_grad()
def evaluate(split, full_preds=False):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    for batch in loaders[split]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        loss_sum += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    acc = correct / total
    if not full_preds:
        return acc, loss_sum / total
    # full prediction run for metrics needing raw sequences
    full_seqs = loaders[split].dataset.raw_seq
    y_true = loaders[split].dataset.y.tolist()
    pred_list = []
    for i in range(0, len(full_seqs), batch_size):
        chunk = full_seqs[i : i + batch_size]
        enc = [encode_sequence(s) for s in chunk]
        lengths = torch.tensor([len(x) for x in enc])
        maxlen = lengths.max()
        inp = torch.full((len(enc), maxlen), vocab[PAD], dtype=torch.long)
        for j, row in enumerate(enc):
            inp[j, : len(row)] = torch.tensor(row)
        logits = model(inp.to(device), lengths.to(device))
        pred_list.extend(logits.argmax(-1).cpu().tolist())
    swa = shape_weighted_accuracy(full_seqs, y_true, pred_list)
    cwa = color_weighted_accuracy(full_seqs, y_true, pred_list)
    return acc, loss_sum / total, swa, cwa, pred_list, y_true, full_seqs


# ---------------------- training w/ early stopping -------------------
max_epochs = 20
patience = 3
best_val_loss = float("inf")
patience_ctr = 0
best_state = None

for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in loaders["train"]:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(datasets["train"])
    train_acc, _ = evaluate("train")
    val_acc, val_loss = evaluate("dev")
    # store logs
    ed = experiment_data["num_epochs"]["SPR_BENCH"]
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train"].append({"acc": train_acc})
    ed["metrics"]["val"].append({"acc": val_acc})
    ed["timestamps"].append(str(datetime.datetime.now()))
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
    )

    # early stopping check
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_ctr = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break

# -------------------- load best model & final eval -------------------
if best_state is not None:
    model.load_state_dict(best_state)

test_acc, _, test_swa, test_cwa, preds, gts, seqs = evaluate("test", full_preds=True)
print(f"TEST  acc={test_acc:.3f}  SWA={test_swa:.3f}  CWA={test_cwa:.3f}")

train_sigs = set(compute_signatures(raw_data["train"]["sequence"]))
test_sigs = compute_signatures(seqs)
novel_idx = [i for i, sg in enumerate(test_sigs) if sg not in train_sigs]
NRGS = (
    sum(1 for i in novel_idx if preds[i] == gts[i]) / len(novel_idx)
    if novel_idx
    else 0.0
)
print(f"Novel Rule Generalization Score (NRGS): {NRGS:.3f}")

# store test metrics
ed = experiment_data["num_epochs"]["SPR_BENCH"]
ed["metrics"]["test"] = {"acc": test_acc, "swa": test_swa, "cwa": test_cwa}
ed["metrics"]["NRGS"] = NRGS
ed["predictions"] = preds
ed["ground_truth"] = gts

# --------------------------- save ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

# -------------------------- plot -------------------------------------
plt.figure()
plt.plot(ed["losses"]["train"], label="train")
plt.plot(ed["losses"]["val"], label="val")
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve_SPR.png"))
plt.close()
