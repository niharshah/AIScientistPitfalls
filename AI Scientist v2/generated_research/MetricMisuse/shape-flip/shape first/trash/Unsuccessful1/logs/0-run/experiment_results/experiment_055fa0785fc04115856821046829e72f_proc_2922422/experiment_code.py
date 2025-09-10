import os, pathlib, time, random, math, numpy as np, torch
from typing import Dict, List
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ experiment tracker ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ------------ device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- helpers ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(
    seqs: List[str], y_true: List[int], y_pred: List[int]
) -> float:
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return float(sum(correct)) / float(sum(weights)) if sum(weights) else 0.0


# -------------- torch dataset -------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], label2idx: Dict[str, int]):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_text = self.seqs[idx]
        tokens = seq_text.split()
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        # simple symbolic features
        symb = [
            count_shape_variety(seq_text),
            len(set(tok[1] for tok in tokens if len(tok) > 1)),
            len(tokens),
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "symbolic": torch.tensor(symb, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": seq_text,
            "length": len(ids),
        }


def build_vocab(sequences: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for s in sequences:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in freq.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def collate_fn(batch):
    # sort by length (requirement for pack_padded_sequence)
    batch.sort(key=lambda x: x["length"], reverse=True)
    max_len = batch[0]["length"]
    input_ids, lengths, symb, labels, texts = [], [], [], [], []

    for b in batch:
        pad_len = max_len - b["length"]
        pad_tensor = torch.zeros(pad_len, dtype=torch.long)
        seq_tensor = torch.cat(
            [b["input_ids"], pad_tensor]
        ).long()  # <-- enforce long dtype
        input_ids.append(seq_tensor)
        lengths.append(b["length"])
        symb.append(b["symbolic"])
        labels.append(b["label"])
        texts.append(b["seq_text"])

    return {
        "input_ids": torch.stack(input_ids, dim=0).long(),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "symbolic": torch.stack(symb, dim=0),
        "label": torch.stack(labels, dim=0),
        "seq_text": texts,
    }


# ------------------ model -----------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.symb_ff = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, emb_dim * 2)
        )
        self.fc = nn.Linear(emb_dim * 4, num_classes)

    def forward(self, ids, lengths, symb_feats):
        ids = ids.to(device)
        lengths = lengths.cpu()  # pack_padded_sequence wants cpu lengths
        symb_feats = symb_feats.to(device)

        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(ids), lengths, batch_first=True, enforce_sorted=True
        )
        _, h = self.gru(packed)  # h: [2, B, H]
        h = torch.cat([h[0], h[1]], dim=-1)  # [B, 2H]
        symb_vec = self.symb_ff(symb_feats)  # [B, 2H]
        combined = torch.cat([h, symb_vec], dim=-1)  # [B, 4H]
        return self.fc(combined)


# -------- training / evaluation ----------
def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_examples, loss_sum = 0, 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"], batch["symbolic"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch["label"].size(0)
        total_examples += batch["label"].size(0)
    return loss_sum / total_examples


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_examples, loss_sum = 0, 0.0
    preds_all, labels_all, seqs_all = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["lengths"], batch["symbolic"])
        loss = loss_fn(logits, batch["label"])

        loss_sum += loss.item() * batch["label"].size(0)
        total_examples += batch["label"].size(0)

        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(batch["label"].cpu().tolist())
        seqs_all.extend(batch["seq_text"])
    avg_loss = loss_sum / total_examples
    return avg_loss, seqs_all, labels_all, preds_all


# ------------- run experiment ------------
def run_experiment(emb_dim: int, epochs: int = 5):
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    label_set = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(label_set)}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    model = NeuralSymbolicClassifier(len(vocab), emb_dim, len(label_set)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, v_seqs, v_true, v_pred = eval_epoch(model, dev_loader, loss_fn)
        swa_val = shape_weighted_accuracy(v_seqs, v_true, v_pred)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(swa_val)
        experiment_data["SPR_BENCH"]["epochs"].append((emb_dim, epoch))

        print(
            f"[emb={emb_dim}] Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}  SWA={swa_val:.4f}"
        )

    # final test
    _, t_seqs, t_true, t_pred = eval_epoch(model, test_loader, loss_fn)
    swa_test = shape_weighted_accuracy(t_seqs, t_true, t_pred)
    print(f"[emb={emb_dim}] TEST SWA={swa_test:.4f}")

    experiment_data["SPR_BENCH"]["metrics"]["test"].append(swa_test)
    experiment_data["SPR_BENCH"]["predictions"].extend(t_pred)
    experiment_data["SPR_BENCH"]["ground_truth"].extend(t_true)


# ------------ launch 2 configs ----------
for dim in [64, 128]:
    run_experiment(dim, epochs=5)
    torch.cuda.empty_cache()

# ----------- persist results ------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
