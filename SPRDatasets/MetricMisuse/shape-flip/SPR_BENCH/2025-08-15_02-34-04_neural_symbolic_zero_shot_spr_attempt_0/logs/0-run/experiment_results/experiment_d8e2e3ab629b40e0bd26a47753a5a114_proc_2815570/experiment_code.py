import os, time, json, math, warnings, pathlib

warnings.filterwarnings("ignore")

# -------------------- working dir --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- basic setup --------------------
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset helpers --------------------
def find_spr_root() -> pathlib.Path:
    env_path = os.getenv("SPR_DIR")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        print(f"Found SPR_BENCH at: {env_path}")
        return pathlib.Path(env_path)
    cwd = pathlib.Path.cwd()
    for p in [cwd / "SPR_BENCH", *cwd.resolve().parents]:
        cand = p / "SPR_BENCH"
        if (cand / "train.csv").exists():
            print(f"Found SPR_BENCH at: {cand}")
            return cand
    raise FileNotFoundError("SPR_BENCH dataset not found.")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split_name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({k: _load(k) for k in ["train", "dev", "test"]})


# ---------- metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# -------------------- Dataset class --------------------
class SPRDataset(Dataset):
    """
    No truncation!  Return raw token-ids list; padding is done in collate_fn.
    """

    def __init__(self, hf_split, token2idx, label2idx):
        self.data = hf_split
        self.tok2id = token2idx
        self.lab2id = label2idx

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        return [
            self.tok2id.get(tok, self.tok2id["<unk>"]) for tok in seq.strip().split()
        ]

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = self.encode(row["sequence"])
        return {
            "input_ids": ids,  # list[int]
            "label": self.lab2id[row["label"]],
            "raw_seq": row["sequence"],
        }


def make_collate_fn(pad_idx):
    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        bs = len(batch)
        input_ids = torch.full((bs, max_len), pad_idx, dtype=torch.long)
        lengths = torch.zeros(bs, dtype=torch.long)
        labels = torch.zeros(bs, dtype=torch.long)

        for i, item in enumerate(batch):
            seq = torch.tensor(item["input_ids"], dtype=torch.long)
            input_ids[i, : len(seq)] = seq
            lengths[i] = len(seq)
            labels[i] = item["label"]
        raw_seq = [item["raw_seq"] for item in batch]
        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "label": labels,
            "raw_seq": raw_seq,
        }

    return collate


# -------------------- model --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_cls, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, n_cls)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)


# -------------------- load data & build vocab --------------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)

specials = ["<pad>", "<unk>"]
vocab_set = set()
for s in spr["train"]["sequence"]:
    vocab_set.update(s.strip().split())
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab_set))}
for i, tok in enumerate(specials):
    token2idx[tok] = i
pad_idx = token2idx["<pad>"]

labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds = SPRDataset(spr["train"], token2idx, label2idx)
dev_ds = SPRDataset(spr["dev"], token2idx, label2idx)
test_ds = SPRDataset(spr["test"], token2idx, label2idx)

collate_fn = make_collate_fn(pad_idx)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

# -------------------- hyperparameter sweep --------------------
embed_dims = [32, 64, 128, 256]
num_epochs = 5
experiment_data = {"embedding_dim": {}}
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, train_flag, optimizer=None):
    model.train() if train_flag else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels, all_seqs = [], [], []
    with torch.set_grad_enabled(train_flag):
        for batch in loader:
            # move tensors to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = total_loss / total
    y_true = [idx2label[i] for i in all_labels]
    y_pred = [idx2label[i] for i in all_preds]
    swa = shape_weighted_accuracy(all_seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(all_seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg_loss, (swa, cwa, hwa), y_true, y_pred


for emb_dim in embed_dims:
    print(f"\n--- Embedding dim: {emb_dim} ---")
    model = GRUClassifier(len(token2idx), emb_dim, 64, len(labels), pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp_entry = {
        "metrics": {"train": [], "val": [], "test": None},
        "losses": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_met, _, _ = run_epoch(model, train_loader, True, optimizer)
        vl_loss, vl_met, _, _ = run_epoch(model, dev_loader, False)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(vl_loss)
        exp_entry["metrics"]["train"].append(tr_met)
        exp_entry["metrics"]["val"].append(vl_met)
        exp_entry["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}: validation_loss = {vl_loss:.4f} "
            f"SWA={vl_met[0]:.4f} CWA={vl_met[1]:.4f} HWA={vl_met[2]:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    tst_loss, tst_met, y_true_tst, y_pred_tst = run_epoch(model, test_loader, False)
    exp_entry["losses"]["test"] = tst_loss
    exp_entry["metrics"]["test"] = tst_met
    exp_entry["predictions"] = y_pred_tst
    exp_entry["ground_truth"] = y_true_tst
    print(f"Test -> SWA={tst_met[0]:.4f}  CWA={tst_met[1]:.4f}  HWA={tst_met[2]:.4f}")

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(exp_entry["losses"]["train"], label="train")
    ax.plot(exp_entry["losses"]["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"SPR GRU Loss (emb={emb_dim})")
    ax.legend()
    plt.savefig(os.path.join(working_dir, f"spr_loss_curve_emb{emb_dim}.png"))
    plt.close(fig)

    experiment_data["embedding_dim"][emb_dim] = exp_entry
    del model
    torch.cuda.empty_cache()

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nAll outputs saved to {working_dir}")
