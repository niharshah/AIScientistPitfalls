import os, pathlib, time, json, math, random
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict  # HF datasets light import

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # we log SWA only
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": {},
        "meta": {},
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------- helper functions (from given snippet) ----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) > 0 else 0.0


# ------------------------ torch Dataset ------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab: Dict[str, int], lbl2idx: Dict[str, int]):
        self.seq_txt = hf_split["sequence"]
        self.labels = [lbl2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq_txt)

    def __getitem__(self, idx):
        tokens = self.seq_txt[idx].split()
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        feat = torch.tensor(
            [
                len(tokens),
                count_shape_variety(self.seq_txt[idx]),
                count_color_variety(self.seq_txt[idx]),
            ],
            dtype=torch.float,
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "sym_feats": feat,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": self.seq_txt[idx],
        }


def build_vocab(train_sequences: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq = {}
    for s in train_sequences:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in freq.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def collate(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    ids, lbls, feats, texts = [], [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        seq_ids = item["input_ids"]
        if pad > 0:
            seq_ids = torch.cat([seq_ids, torch.zeros(pad, dtype=torch.long)])
        ids.append(seq_ids)
        lbls.append(item["label"])
        feats.append(item["sym_feats"])
        texts.append(item["seq_text"])
    return {
        "input_ids": torch.stack(ids),
        "sym_feats": torch.stack(feats),
        "label": torch.stack(lbls),
        "seq_text": texts,
    }


# ----------------------------- model ---------------------------------------------
class HybridClassifier(nn.Module):
    def __init__(
        self,
        vocab_sz: int,
        emb_dim: int,
        hid: int,
        nclass: int,
        use_symbolic: bool = True,
    ):
        super().__init__()
        self.use_symbolic = use_symbolic
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        rnn_out = hid * 2
        sym_dim = 3
        if use_symbolic:
            self.sym_ff = nn.Sequential(
                nn.Linear(sym_dim, 8), nn.ReLU(), nn.Linear(8, 8)
            )
            total = rnn_out + 8
        else:
            total = rnn_out
        self.classifier = nn.Linear(total, nclass)

    def forward(self, ids, sym_feats=None):
        mask = ids != 0
        emb = self.emb(ids)
        lengths = mask.sum(1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=True
        )
        _, h = self.rnn(packed)  # h: (2, B, hid)
        sent_vec = torch.cat([h[0], h[1]], dim=1)  # (B, 2*hid)
        if self.use_symbolic:
            sym_vec = self.sym_ff(sym_feats)
            sent_vec = torch.cat([sent_vec, sym_vec], dim=1)
        return self.classifier(sent_vec)


# ------------------------- training / evaluation -------------------------------
def train_epoch(model, loader, optim, crit):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optim.zero_grad()
        out = model(batch["input_ids"], batch["sym_feats"])
        loss = crit(out, batch["label"])
        loss.backward()
        optim.step()
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, crit):
    model.eval()
    total = 0.0
    all_pred, all_lbl, all_seq = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch["input_ids"], batch["sym_feats"])
        loss = crit(out, batch["label"])
        total += loss.item() * batch["label"].size(0)
        preds = out.argmax(-1).cpu().tolist()
        all_pred.extend(preds)
        all_lbl.extend(batch["label"].cpu().tolist())
        all_seq.extend(batch["seq_text"])
    return total / len(loader.dataset), all_seq, all_lbl, all_pred


# ----------------------------- main experiment --------------------------------
def run_variant(use_symbolic=True, tag="hybrid"):
    DATA_ROOT = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    spr = load_spr_bench(DATA_ROOT)

    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    lbl2idx = {l: i for i, l in enumerate(labels)}

    tr_ds = SPRTorchDataset(spr["train"], vocab, lbl2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, lbl2idx)
    te_ds = SPRTorchDataset(spr["test"], vocab, lbl2idx)

    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=256, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=256, collate_fn=collate)

    model = HybridClassifier(
        len(vocab), 128, 128, len(labels), use_symbolic=use_symbolic
    ).to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 7
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, tr_loader, optim, crit)
        val_loss, seqs, y_t, y_p = eval_epoch(model, dev_loader, crit)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(swa)
        print(f"({tag}) Epoch {epoch}: val_loss={val_loss:.4f}  SWA={swa:.4f}")

    # final test
    _, seqs_t, y_tt, y_pt = eval_epoch(model, te_loader, crit)
    swa_test = shape_weighted_accuracy(seqs_t, y_tt, y_pt)
    experiment_data["SPR_BENCH"]["predictions"][tag] = y_pt
    experiment_data["SPR_BENCH"]["ground_truth"][tag] = y_tt
    experiment_data["SPR_BENCH"]["meta"][f"SWA_test_{tag}"] = swa_test
    print(f"({tag}) TEST SWA={swa_test:.4f}")

    # cleanup
    del model
    torch.cuda.empty_cache()


# run both variants
run_variant(use_symbolic=False, tag="neural_only")
run_variant(use_symbolic=True, tag="hybrid")

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
