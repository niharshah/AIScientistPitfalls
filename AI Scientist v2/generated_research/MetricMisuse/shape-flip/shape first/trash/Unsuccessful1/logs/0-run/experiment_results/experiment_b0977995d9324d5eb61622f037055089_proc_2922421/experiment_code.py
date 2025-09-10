import os, pathlib, math, random, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- mandatory working directory / device --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # SWA only
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": {},
        "meta": {},
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------- helper utilities ------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(1, sum(weights))


# ----------------------- Dataset / Vocab ----------------------------------------
def build_vocab(seqs, min_freq=1):
    freq = {}
    for s in seqs:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for t, c in freq.items():
        if c >= min_freq:
            vocab[t] = len(vocab)
    return vocab


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, lbl2idx):
        self.seqs = hf_split["sequence"]
        self.lbl = [lbl2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.vocab.get(tok, 1) for tok in seq.split()]
        sym = torch.tensor(
            [
                len(ids),
                count_shape_variety(seq),
                len(set(tok[1] for tok in seq.split())),
            ],
            dtype=torch.float,
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "sym": sym,
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "seq_text": seq,
        }


def collate(batch, max_len=60):
    L = max(min(max(len(b["input_ids"]) for b in batch), max_len), 1)
    pad_id = 0
    ids = []
    for b in batch:
        arr = b["input_ids"][:L]
        if len(arr) < L:
            arr = torch.cat([arr, torch.full((L - len(arr),), pad_id)])
        ids.append(arr)
    return {
        "input_ids": torch.stack(ids),
        "sym": torch.stack([b["sym"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "seq_text": [b["seq_text"] for b in batch],
    }


# ------------------------------ Model -------------------------------------------
class FiLMTransformer(nn.Module):
    def __init__(self, vocab_sz, d_model=128, nhead=4, nlayers=2, nclass=10):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(64, d_model)  # max 64 tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        # FiLM from symbolic feats
        self.film = nn.Sequential(nn.Linear(3, d_model), nn.Tanh())
        self.cls = nn.Linear(d_model, nclass)

    def forward(self, ids, sym):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.encoder(x)
        cls_vec = x[:, 0]  # use first token
        gamma = torch.sigmoid(self.film(sym))  # scale in [0,1]
        cls_vec = cls_vec * gamma
        return self.cls(cls_vec)


# ---------------------- Train / Eval loops --------------------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        opt.zero_grad()
        out = model(batch["input_ids"], batch["sym"])
        loss = loss_fn(out, batch["label"])
        loss.backward()
        opt.step()
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total = 0
    all_pred, all_lbl, all_seq = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["input_ids"], batch["sym"])
        loss = loss_fn(out, batch["label"])
        total += loss.item() * batch["label"].size(0)
        p = out.argmax(-1).cpu().tolist()
        all_pred.extend(p)
        all_lbl.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    swa = shape_weighted_accuracy(all_seq, all_lbl, all_pred)
    return total / len(loader.dataset), swa, all_seq, all_lbl, all_pred


# ------------------------------ Experiment --------------------------------------
def run_experiment():
    data_root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    spr = load_spr_bench(data_root)
    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    lbl2idx = {l: i for i, l in enumerate(labels)}
    # datasets
    tr_ds = SPRTorchDataset(spr["train"], vocab, lbl2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, lbl2idx)
    te_ds = SPRTorchDataset(spr["test"], vocab, lbl2idx)
    # loaders
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=256, collate_fn=collate)
    # model
    model = FiLMTransformer(len(vocab), nclass=len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    epochs = 6
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, tr_loader, opt, loss_fn)
        v_loss, v_swa, *_ = eval_epoch(model, dev_loader, loss_fn)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(v_swa)
        print(f"Epoch {ep}: validation_loss = {v_loss:.4f}  SWA = {v_swa:.4f}")
    # final test
    _, t_swa, seqs, gts, preds = eval_epoch(model, te_loader, loss_fn)
    experiment_data["SPR_BENCH"]["predictions"]["FiLM"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"]["FiLM"] = gts
    experiment_data["SPR_BENCH"]["meta"]["SWA_test_FiLM"] = t_swa
    print(f"TEST SWA = {t_swa:.4f}")


run_experiment()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
