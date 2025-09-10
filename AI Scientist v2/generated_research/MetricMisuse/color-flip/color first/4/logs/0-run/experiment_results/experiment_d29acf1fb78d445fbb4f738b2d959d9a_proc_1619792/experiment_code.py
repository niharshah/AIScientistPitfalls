import os, pathlib, random, string, time, math, json, numpy as np, torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# GPU / CPU handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


######################################################################
# ---------- DATA LOADING (real or synthetic) ------------------------
######################################################################
def try_load_real_spr(path: pathlib.Path):
    try:
        # informal existence check
        if (path / "train.csv").exists():
            import datasets
            from datasets import DatasetDict

            def _load(splitfile):
                return datasets.load_dataset(
                    "csv",
                    data_files=str(path / splitfile),
                    split="train",
                    cache_dir=".cache_dsets",
                )

            d = DatasetDict()
            d["train"] = _load("train.csv")
            d["dev"] = _load("dev.csv")
            d["test"] = _load("test.csv")
            return d
    except Exception as e:
        print("Real SPR_BENCH not found or failed to load:", e)
    return None


def make_toy_spr(
    num_train=1000,
    num_dev=200,
    num_test=200,
    max_len=8,
    shapes=list("ABCD"),
    colors=list("1234"),
):
    """
    Creates a toy SPR-like dataset where the label is 0
    if the first token's shape is in the first half of shapes, else 1.
    """

    def _gen(n):
        rows = []
        for i in range(n):
            seqlen = random.randint(3, max_len)
            seq_tokens = []
            for _ in range(seqlen):
                s = random.choice(shapes)
                c = random.choice(colors)
                seq_tokens.append(f"{s}{c}")
            sequence = " ".join(seq_tokens)
            label = 0 if seq_tokens[0][0] in shapes[: len(shapes) // 2] else 1
            rows.append({"id": i, "sequence": sequence, "label": label})
        return rows

    d = {"train": _gen(num_train), "dev": _gen(num_dev), "test": _gen(num_test)}
    # wrap to mimic HF dataset interface (list-like is enough for this script)
    return {k: d[k] for k in d}


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dataset = try_load_real_spr(DATA_PATH)
if dataset is None:
    print("Falling back to synthetic SPR dataset")
    dataset = make_toy_spr()


######################################################################
# ---------- METRIC HELPERS -----------------------------------------
######################################################################
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pattern_complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


######################################################################
# ---------- VOCABULARY ---------------------------------------------
######################################################################
PAD, UNK = "<pad>", "<unk>"


def build_vocab(train_rows):
    vocab = {PAD: 0, UNK: 1}
    for row in train_rows:
        for tok in row["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(dataset["train"])
print(f"Vocab size: {len(vocab)}")


def encode_sequence(seq, vocab):
    return torch.tensor(
        [vocab.get(tok, vocab[UNK]) for tok in seq.split()], dtype=torch.long
    )


######################################################################
# ---------- DATASET & DATALOADER ------------------------------------
######################################################################
class SPRDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "id": r["id"],
            "sequence_str": r["sequence"],
            "input_ids": encode_sequence(r["sequence"], self.vocab),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


def collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    padded = pad_sequence(input_ids, batch_first=True, padding_value=vocab[PAD])
    mask = (padded != vocab[PAD]).float()
    labels = torch.stack([item["label"] for item in batch])
    seq_strs = [item["sequence_str"] for item in batch]
    return {
        "input_ids": padded,
        "mask": mask,
        "labels": labels,
        "seq_strs": seq_strs,
        "ids": [item["id"] for item in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRDataset(dataset["train"], vocab),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    SPRDataset(dataset["dev"], vocab),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate,
)


######################################################################
# ---------- MODEL ---------------------------------------------------
######################################################################
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, mask):
        emb = self.embed(input_ids)  # (B,L,E)
        packed_out, _ = self.gru(emb)  # (B,L,H*2)
        # mean pooling with mask
        summed = (packed_out * mask.unsqueeze(-1)).sum(1)
        lens = mask.sum(1).clamp(min=1)
        mean = summed / lens.unsqueeze(-1)
        logits = self.fc(mean)
        return logits


model = BiGRUClassifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

######################################################################
# ---------- EXPERIMENT DATA STORAGE --------------------------------
######################################################################
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


######################################################################
# ---------- TRAINING LOOP ------------------------------------------
######################################################################
def run_epoch(model, loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_seqs, all_true, all_pred = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            # move tensors
            batch_tensors = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            input_ids = batch_tensors["input_ids"]
            mask = batch_tensors["mask"]
            labels = batch_tensors["labels"]
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(-1).detach().cpu().tolist()
            total_correct += sum([p == t for p, t in zip(preds, labels.cpu().tolist())])
            total += labels.size(0)
            # store for metrics
            all_seqs.extend(batch["seq_strs"])
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds)
    avg_loss = total_loss / total
    acc = total_correct / total
    pcwa = pattern_complexity_weighted_accuracy(all_seqs, all_true, all_pred)
    return avg_loss, acc, pcwa, all_pred, all_true


epochs = 5
for epoch in range(1, epochs + 1):
    tr_loss, tr_acc, tr_pcwa, _, _ = run_epoch(model, train_loader, train=True)
    val_loss, val_acc, val_pcwa, val_preds, val_true = run_epoch(
        model, dev_loader, train=False
    )

    experiment_data["SPR"]["metrics"]["train"].append(
        {"epoch": epoch, "accuracy": tr_acc, "pcwa": tr_pcwa}
    )
    experiment_data["SPR"]["metrics"]["val"].append(
        {"epoch": epoch, "accuracy": val_acc, "pcwa": val_pcwa}
    )
    experiment_data["SPR"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["predictions"].append(val_preds)
    experiment_data["SPR"]["ground_truth"].append(val_true)
    experiment_data["SPR"]["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.3f}, val_PCWA={val_pcwa:.3f}"
    )

######################################################################
# ---------- SAVE RESULTS -------------------------------------------
######################################################################
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))

# optional simple plot
try:
    import matplotlib.pyplot as plt

    epochs_range = range(1, epochs + 1)
    val_pcwa = [m["pcwa"] for m in experiment_data["SPR"]["metrics"]["val"]]
    plt.figure()
    plt.plot(epochs_range, val_pcwa, marker="o")
    plt.title("Validation PCWA over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PCWA")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "SPR_val_pcwa.png"))
    print("Figure saved.")
except Exception as e:
    print("Plotting skipped:", e)
