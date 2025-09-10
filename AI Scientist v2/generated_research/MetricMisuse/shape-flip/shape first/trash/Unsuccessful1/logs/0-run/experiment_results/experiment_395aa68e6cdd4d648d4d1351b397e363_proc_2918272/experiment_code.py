import os, pathlib, random, time, json
import numpy as np
import torch
from typing import List, Dict
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# basic setup & reproducibility -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------
# dataset helpers (same logic as original) -------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _load(f"{split}.csv")
    return out


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split()))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, yt, yp):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, yt, yp)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, yt, yp):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, yt, yp)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(seqs, yt, yp):
    swa = shape_weighted_accuracy(seqs, yt, yp)
    cwa = color_weighted_accuracy(seqs, yt, yp)
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ------------------------------------------------------------------
# torch dataset & utilities ----------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab, label2idx):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [
            self.vocab.get(tok, self.vocab["<unk>"]) for tok in self.seqs[idx].split()
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_text": self.seqs[idx],
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


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])
    pad_id = 0
    padded = [
        torch.cat(
            [
                b["input_ids"],
                torch.full((max_len - len(b["input_ids"]),), pad_id, dtype=torch.long),
            ]
        )
        for b in batch
    ]
    return {
        "input_ids": torch.stack(padded),
        "label": torch.tensor([b["label"] for b in batch]),
        "seq_text": [b["seq_text"] for b in batch],
    }


# ------------------------------------------------------------------
class BagOfTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_cls: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        emb = self.emb(x) * mask
        mean = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(mean)


# ------------------------------------------------------------------
def train_epoch(model, loader, opt, crit):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        loss.backward()
        opt.step()
        total += loss.item() * batch["label"].size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total = 0.0
    all_p = []
    all_l = []
    all_s = []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = crit(logits, batch["label"])
        total += loss.item() * batch["label"].size(0)
        preds = logits.argmax(-1).cpu().tolist()
        all_p.extend(preds)
        all_l.extend(batch["label"].cpu().tolist())
        all_s.extend(batch["seq_text"])
    swa = shape_weighted_accuracy(all_s, all_l, all_p)
    cwa = color_weighted_accuracy(all_s, all_l, all_p)
    hwa = harmonic_weighted_accuracy(all_s, all_l, all_p)
    return total / len(loader.dataset), swa, cwa, hwa, all_s, all_l, all_p


# ------------------------------------------------------------------
def run_experiment(
    train_loader,
    dev_loader,
    test_loader,
    vocab_size: int,
    num_cls: int,
    epochs: int = 30,
    patience: int = 3,
    lr: float = 1e-3,
):
    model = BagOfTokenClassifier(vocab_size, 64, num_cls).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_hwa = -1.0
    best_state = None
    no_improve = 0
    logs = {
        "train_loss": [],
        "val_loss": [],
        "val_SWA": [],
        "val_CWA": [],
        "val_HWA": [],
    }

    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, crit)
        val_loss, swa, cwa, hwa, _, _, _ = evaluate(model, dev_loader, crit)
        logs["train_loss"].append(tr_loss)
        logs["val_loss"].append(val_loss)
        logs["val_SWA"].append(swa)
        logs["val_CWA"].append(cwa)
        logs["val_HWA"].append(hwa)
        print(
            f"Epoch {ep}/{epochs} - train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
            f"SWA {swa:.4f} CWA {cwa:.4f} HWA {hwa:.4f}"
        )
        # ---------- EARLY-STOPPING NOW USES HWA (MAX) -------------
        if hwa > best_hwa:
            best_hwa = hwa
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered – no HWA improvement.")
            break

    # reload best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_loss, swa_t, cwa_t, hwa_t, seqs_t, y_tt, y_pp = evaluate(
        model, test_loader, crit
    )
    print(f"TEST RESULTS – SWA {swa_t:.4f} | CWA {cwa_t:.4f} | HWA {hwa_t:.4f}")

    return logs, {"SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}, y_pp, y_tt


# ------------------------------------------------------------------
def main():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)

    vocab = build_vocab(spr["train"]["sequence"])
    labels = sorted(set(spr["train"]["label"]))
    label2idx = {l: i for i, l in enumerate(labels)}

    train_ds = SPRTorchDataset(spr["train"], vocab, label2idx)
    dev_ds = SPRTorchDataset(spr["dev"], vocab, label2idx)
    test_ds = SPRTorchDataset(spr["test"], vocab, label2idx)

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )

    epoch_grid = [10, 20, 30]
    experiment_data = {"num_epochs_tuning": {"SPR_BENCH": {}}}

    for n_ep in epoch_grid:
        print(
            "=" * 70
            + f"\nTraining for {n_ep} epochs (patience on HWA={3})\n"
            + "=" * 70
        )
        logs, metrics_test, y_pred, y_gold = run_experiment(
            train_loader,
            dev_loader,
            test_loader,
            vocab_size=len(vocab),
            num_cls=len(labels),
            epochs=n_ep,
            patience=3,
            lr=1e-3,
        )
        exp_entry = {
            "losses": logs,
            "metrics": {"test": metrics_test},
            "predictions": y_pred,
            "ground_truth": y_gold,
        }
        experiment_data["num_epochs_tuning"]["SPR_BENCH"][f"epochs_{n_ep}"] = exp_entry
        # save intermediate result after each run
        np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# run immediately
main()
