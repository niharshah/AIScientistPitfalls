import os, pathlib, random, time, json, math
import numpy as np, torch
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ----------------- reproducibility -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- working dir ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- experiment container ------------
experiment_data = {
    "num_epochs_tuning": {
        "SPR_BENCH": {}  # will be filled with one entry per epoch-setting
    }
}

# ----------------- device --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------- locate SPR_BENCH ----------------
def find_spr_bench_path() -> pathlib.Path:
    cand = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cand:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print("Found SPR_BENCH at:", p)
            return p
    raise FileNotFoundError("SPR_BENCH with train/dev/test csv not found.")


DATA_PATH = find_spr_bench_path()


# ----------------- dataset utils -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred))
    return num / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    num = sum(w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred))
    return num / sum(w) if sum(w) > 0 else 0.0


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ----------------- vocab / labels ------------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labels = sorted({ex["label"] for ex in dataset})
    return {lab: i for i, lab in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print(f"Vocab={len(vocab)}, num_labels={num_labels}")


# ----------------- torch dataset -------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dset, vocab, label2id):
        self.data = hf_dset
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    seqs = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = b["input_ids"]
        labels[i] = b["label"]
        seqs.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": seqs}


train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)


# ----------------- model ---------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        mean = summed / lengths
        return self.fc(mean)


# ----------------- training util -------------------
def run_training(max_epochs: int, patience: int = 3):
    model = BiLSTMClassifier(len(vocab), 64, 128, num_labels, pad_idx=pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    logs = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    best_hwa = -1
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * batch["labels"].size(0)
        train_loss = run_loss / len(train_ds)
        logs["losses"]["train"].append(train_loss)
        logs["metrics"]["train"].append({"epoch": epoch, "loss": train_loss})
        # ---- eval ----
        model.eval()
        val_loss = 0.0
        all_pred = []
        all_true = []
        all_seq = []
        with torch.no_grad():
            for batch in dev_loader:
                t_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(t_batch["input_ids"])
                loss = criterion(logits, t_batch["labels"])
                val_loss += loss.item() * t_batch["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                truths = t_batch["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(truths)
                all_seq.extend(batch["sequences"])
        val_loss /= len(dev_ds)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        logs["losses"]["val"].append(val_loss)
        logs["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": val_loss}
        )
        print(
            f"[{max_epochs}e] Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} HWA={hwa:.4f}"
        )
        # early stopping on HWA
        if hwa > best_hwa + 1e-5:
            best_hwa = hwa
            best_state = model.state_dict()
            epochs_no_improve = 0
            logs["predictions"] = all_pred
            logs["ground_truth"] = all_true
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    if best_state is not None:  # load best for reproducibility
        model.load_state_dict(best_state)
    return logs


# ----------------- hyperparameter tuning -----------
epoch_grid = [5, 10, 20, 30]
for epochs in epoch_grid:
    print(f"\n=== Training for max {epochs} epochs ===")
    logs = run_training(epochs, patience=3)
    experiment_data["num_epochs_tuning"]["SPR_BENCH"][str(epochs)] = logs

# ----------------- save ----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
