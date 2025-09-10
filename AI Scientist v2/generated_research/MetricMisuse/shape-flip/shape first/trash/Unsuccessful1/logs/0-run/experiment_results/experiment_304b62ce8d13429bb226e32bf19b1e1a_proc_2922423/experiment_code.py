import os, pathlib, time, json, math, random
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- GPU / CPU ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- SPR utilities (from given snippet) ------------------
from datasets import load_dataset, DatasetDict


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# ---------------- Torch dataset --------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], label2idx: Dict[str, int]):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        tokens = seq.split()
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        sym_feats = torch.tensor(
            [len(tokens), count_shape_variety(seq), count_color_variety(seq)],
            dtype=torch.float,
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "sym_feats": sym_feats,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": seq,
        }


def build_vocab(train_sequences: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for s in train_sequences:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, cnt in freq.items():
        if cnt >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    input_ids, labels, sym, seq_text = [], [], [], []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(item["label"])
        sym.append(item["sym_feats"])
        seq_text.append(item["seq_text"])
    return {
        "input_ids": torch.stack(input_ids),
        "sym_feats": torch.stack(sym),
        "label": torch.tensor(labels, dtype=torch.long),
        "seq_text": seq_text,
    }


# ---------------- Neural-Symbolic model ------------------------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, emb_dim: int, sym_feat_dim: int, num_classes: int
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_feat_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.fc = nn.Linear(emb_dim + 32, num_classes)

    def forward(self, ids, sym_feats):
        mask = (ids != 0).unsqueeze(-1)
        emb = self.emb(ids) * mask
        mean = emb.sum(1) / mask.sum(1).clamp(min=1)
        sym_vec = self.sym_mlp(sym_feats)
        concat = torch.cat([mean, sym_vec], dim=-1)
        return self.fc(concat)


# ---------------- Training / Eval loops ------------------------------
def train_epoch(model, loader, optim, criterion):
    model.train()
    tot_loss = 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optim.step()
        tot_loss += loss.item() * batch["label"].size(0)
    return tot_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tot_loss, seqs, ys, preds = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = criterion(logits, batch["label"])
        tot_loss += loss.item() * batch["label"].size(0)
        pred = logits.argmax(-1).cpu().tolist()
        preds.extend(pred)
        ys.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["seq_text"])
    return tot_loss / len(loader.dataset), seqs, ys, preds


# ---------------- Main experiment ------------------------------------
def run_experiment():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate_fn)

    model = NeuralSymbolicClassifier(len(vocab), 256, 3, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, criterion)
        val_loss, seqs, y_true, y_pred = eval_epoch(model, dev_loader, criterion)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(swa)

        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  SWA = {swa:.4f}")

    # final test
    _, seqs_t, y_true_t, y_pred_t = eval_epoch(model, test_loader, criterion)
    swa_test = shape_weighted_accuracy(seqs_t, y_true_t, y_pred_t)
    experiment_data["SPR_BENCH"]["metrics"]["test"] = swa_test
    experiment_data["SPR_BENCH"]["predictions"] = y_pred_t
    experiment_data["SPR_BENCH"]["ground_truth"] = y_true_t
    print(f"TEST SWA = {swa_test:.4f}")

    # save results
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

    # free memory
    del model
    torch.cuda.empty_cache()


run_experiment()
