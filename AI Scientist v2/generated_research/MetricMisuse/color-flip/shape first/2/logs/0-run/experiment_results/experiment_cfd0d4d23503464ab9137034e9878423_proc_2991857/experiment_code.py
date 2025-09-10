import os, pathlib, random, time, json, math, gc
import torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- reproducibility ----------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment container ----------
experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {"runs": {}}  # each weight_decay value gets its own dict
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    candidates = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p}")
            return p
    raise FileNotFoundError(
        "SPR_BENCH directory with train.csv/dev.csv/test.csv not found."
    )


DATA_PATH = find_spr_bench_path()


# ---------- dataset utilities ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------- load dataset ----------
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- build vocab / labels ----------
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
num_labels, pad_id = len(label2id), vocab["<pad>"]
print(f"Vocab size = {len(vocab)}, num_labels = {num_labels}")


# ---------- Torch dataset ----------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab, label2id):
        self.data, self.vocab, self.label2id = hf_dataset, vocab, label2id

    def __len__(self):
        return len(self.data)

    def encode_seq(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(
                self.encode_seq(ex["sequence"]), dtype=torch.long
            ),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels, sequences = torch.empty(len(batch), dtype=torch.long), []
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
        labels[i] = b["label"]
        sequences.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": sequences}


train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
)
dev_loader = DataLoader(
    dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
)


# ---------- model definition ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.emb(x)
        outputs, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        mean = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)


# ---------- hyperparameter tuning ----------
weight_decays = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
epochs = 5

for wd in weight_decays:
    print(f"\n===== Training with weight_decay={wd} =====")
    run_key = str(wd)
    experiment_data["weight_decay"]["SPR_BENCH"]["runs"][run_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = BiLSTMClassifier(len(vocab), 64, 128, num_labels, pad_idx=pad_id).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item() * batch["labels"].size(0)
        train_loss = running_loss / len(train_ds)

        # ---- eval ----
        model.eval()
        val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
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

        # ---- log ----
        run_dict = experiment_data["weight_decay"]["SPR_BENCH"]["runs"][run_key]
        run_dict["losses"]["train"].append(train_loss)
        run_dict["losses"]["val"].append(val_loss)
        run_dict["metrics"]["train"].append({"epoch": epoch, "loss": train_loss})
        run_dict["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "hwa": hwa, "loss": val_loss}
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

    # store preds of last epoch
    run_dict["predictions"] = all_pred
    run_dict["ground_truth"] = all_true

    # free memory
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
