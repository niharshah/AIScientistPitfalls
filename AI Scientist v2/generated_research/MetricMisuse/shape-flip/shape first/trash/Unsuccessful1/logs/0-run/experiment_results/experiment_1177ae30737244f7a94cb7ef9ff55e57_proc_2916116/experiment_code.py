import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict
from typing import List
import datetime

# ---------------------------------------------------------------------------
# GPU / device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ---------------------------------------------------------------------------
# ------------------ SPR utility functions  (from provided stub) -------------
from datasets import load_dataset


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(seq) for seq in sequences]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) > 0 else 0.0


# ---------------------------------------------------------------------------


# ------------------------ Dataset ------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_dataset, vocab: dict, label2id: dict, build_vocab=False):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]
        self.ids = hf_dataset["id"]
        self.vocab = vocab
        self.label2id = label2id
        if build_vocab:
            self._build_vocab()
        self.unk = self.vocab.get("<UNK>", 1)

    def _build_vocab(self):
        for seq in self.seqs:
            for tok in seq.strip().split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        toks = seq.strip().split()
        ids = [self.vocab.get(t, self.unk) for t in toks]
        label = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_raw": seq,
        }


def collate_fn(batch):
    # Pad sequences
    lengths = [len(item["input_ids"]) for item in batch]
    max_len = max(lengths)
    padded = []
    for item in batch:
        ids = item["input_ids"]
        if len(ids) < max_len:
            ids = torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)])
        padded.append(ids)
    input_ids = torch.stack(padded)
    labels = torch.stack([item["label"] for item in batch])
    seq_raw = [item["seq_raw"] for item in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "seq_raw": seq_raw,
        "lengths": lengths,
    }


# ------------------------ Model -------------------------------------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        # pack sequences for GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        logits = self.fc(hidden.squeeze(0))
        return logits


# ------------------------ Training utils ----------------------------------
def compute_metrics(seqs: List[str], y_true: List[int], y_pred: List[int]):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = 0.0
    if swa + cwa > 0:
        hwa = 2 * swa * cwa / (swa + cwa)
    return swa, cwa, hwa


# ------------------------ Main experiment ---------------------------------
def run_experiment():
    # ------------------ Load dataset ------------------
    data_path_env = os.environ.get("SPR_DATA_PATH", "./SPR_BENCH")
    data_path = pathlib.Path(data_path_env)
    if not data_path.exists():
        # create tiny synthetic data as fallback
        print("WARNING: SPR_BENCH not found, creating synthetic toy dataset.")
        data_path.mkdir(parents=True, exist_ok=True)

        def synth(split, n):
            ids, seqs, labels = [], [], []
            shapes = ["a", "b", "c"]
            colors = ["1", "2", "3"]
            for i in range(n):
                ln = np.random.randint(3, 8)
                tokens = []
                for _ in range(ln):
                    tokens.append(np.random.choice(shapes) + np.random.choice(colors))
                ids.append(f"{split}-{i}")
                seqs.append(" ".join(tokens))
                labels.append(np.random.randint(0, 2))
            import csv

            with open(data_path / f"{split}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "sequence", "label"])
                for id_, s, l in zip(ids, seqs, labels):
                    writer.writerow([id_, s, l])

        synth("train", 1000)
        synth("dev", 200)
        synth("test", 200)

    spr = load_spr_bench(data_path)
    # ------------------ Build vocab + label map --------------
    vocab = {"<PAD>": 0, "<UNK>": 1}
    # map labels to ids
    all_labels = list(set(spr["train"]["label"]))
    label2id = {lab: idx for idx, lab in enumerate(sorted(all_labels))}
    id2label = {v: k for k, v in label2id.items()}

    train_dataset = SPRDataset(spr["train"], vocab, label2id, build_vocab=True)
    dev_dataset = SPRDataset(spr["dev"], vocab, label2id, build_vocab=False)
    test_dataset = SPRDataset(spr["test"], vocab, label2id, build_vocab=False)

    vocab_size = len(vocab)
    num_classes = len(label2id)
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ------------------ Model, loss, optimizer --------------
    embed_dim = 64
    hidden_dim = 128
    model = SPRClassifier(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------ Experiment data dict ---------------
    experiment_data = {
        "spr_bench": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    # ------------------ Training loop ----------------------
    epochs = 3
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        avg_train_loss = total_loss / len(train_dataset)
        experiment_data["spr_bench"]["losses"]["train"].append(avg_train_loss)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        all_seq = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["lengths"]
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=-1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(labels.cpu().numpy().tolist())
                all_seq.extend(batch["seq_raw"])
        avg_val_loss = val_loss / len(dev_dataset)
        swa, cwa, hwa = compute_metrics(all_seq, y_true, y_pred)
        experiment_data["spr_bench"]["losses"]["val"].append(avg_val_loss)
        experiment_data["spr_bench"]["metrics"]["val"].append(
            {
                "swa": swa,
                "cwa": cwa,
                "hwa": hwa,
                "epoch": epoch,
                "timestamp": str(datetime.datetime.now()),
            }
        )
        print(
            f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}, SWA={swa:.3f}, CWA={cwa:.3f}, HWA={hwa:.3f}"
        )

    # ------------------ Test evaluation --------------------
    model.eval()
    all_seq = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]
            logits = model(input_ids, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            all_seq.extend(batch["seq_raw"])
    swa, cwa, hwa = compute_metrics(all_seq, y_true, y_pred)
    print(f"TEST: SWA={swa:.3f}, CWA={cwa:.3f}, HWA={hwa:.3f}")

    experiment_data["spr_bench"]["predictions"] = y_pred
    experiment_data["spr_bench"]["ground_truth"] = y_true
    experiment_data["spr_bench"]["metrics"]["test"] = {
        "swa": swa,
        "cwa": cwa,
        "hwa": hwa,
    }

    # ------------------ Save experiment data ---------------
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"Saved experiment data to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


run_experiment()
