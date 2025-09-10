# frozen_gru_ablation.py
import os, pathlib, time, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# reproducibility
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- dataset helpers -----------------------------
def find_spr_root() -> pathlib.Path:
    cand = os.getenv("SPR_DIR")
    if cand and (pathlib.Path(cand) / "train.csv").exists():
        return pathlib.Path(cand)
    for p in [pathlib.Path.cwd() / "SPR_BENCH", *pathlib.Path.cwd().resolve().parents]:
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError(
        "SPR_BENCH dataset not found. Set $SPR_DIR or place in cwd/parent."
    )


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# -------------------- metrics -------------------------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------------------- dataset class -------------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, tok2id, lab2id, max_len=30):
        self.data = hf_split
        self.tok2id = tok2id
        self.lab2id = lab2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        ids = [self.tok2id.get(t, self.tok2id["<unk>"]) for t in seq.strip().split()][
            : self.max_len
        ]
        pad = self.max_len - len(ids)
        return ids + [self.tok2id["<pad>"]] * pad, len(ids)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids, real_len = self.encode(row["sequence"])
        return {
            "input_ids": torch.tensor(ids),
            "lengths": torch.tensor(real_len),
            "label": torch.tensor(self.lab2id[row["label"]]),
            "raw_seq": row["sequence"],
        }


# -------------------- model ---------------------------------------
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


# -------------------- prepare data --------------------------------
spr_root = find_spr_root()
spr = load_spr_bench(spr_root)
specials = ["<pad>", "<unk>"]
vocab = set()
for s in spr["train"]["sequence"]:
    vocab.update(s.strip().split())
token2idx = {tok: i + len(specials) for i, tok in enumerate(sorted(vocab))}
for i, t in enumerate(specials):
    token2idx[t] = i
pad_idx = token2idx["<pad>"]

labels = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}

train_ds, dev_ds, test_ds = [
    SPRDataset(spr[spl], token2idx, label2idx) for spl in ("train", "dev", "test")
]
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


# -------------------- training helper -----------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss = tot = 0
    preds, labels_, seqs = [], [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            tot += bs
            pr = logits.argmax(1).cpu().numpy()
            preds.extend(pr)
            labels_.extend(batch["label"].cpu().numpy())
            seqs.extend(batch["raw_seq"])
    avg = tot_loss / tot
    y_true = [idx2label[i] for i in labels_]
    y_pred = [idx2label[i] for i in preds]
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
    return avg, (swa, cwa, hwa), y_true, y_pred


# -------------------- ablation: frozen embedding ------------------
epoch_options = [5, 10, 20, 30]
patience = 3
experiment_data = {"frozen_embeddings": {"SPR_BENCH": {"num_epochs": {}}}}

for num_epochs in epoch_options:
    print(f"\n=== Frozen-Emb: num_epochs={num_epochs} ===")
    torch.cuda.empty_cache()
    model = GRUClassifier(len(token2idx), 32, 64, len(labels), pad_idx).to(device)
    # freeze embedding
    for p in model.emb.parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    run_data = {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": [], "test": None},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    best_hwa = -1
    no_improve = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t = time.time()
        tr_loss, tr_met, *_ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_met, *_ = run_epoch(model, dev_loader, criterion)
        run_data["losses"]["train"].append(tr_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["train"].append(tr_met)
        run_data["metrics"]["val"].append(val_met)
        run_data["timestamps"].append(time.time())
        print(
            f"Epoch {epoch}/{num_epochs} val_loss={val_loss:.4f} "
            f"SWA={val_met[0]:.4f} CWA={val_met[1]:.4f} HWA={val_met[2]:.4f} "
            f"({time.time()-t:.1f}s)"
        )
        if val_met[2] > best_hwa:
            best_hwa = val_met[2]
            no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    test_loss, test_met, y_true_test, y_pred_test = run_epoch(
        model, test_loader, criterion
    )
    run_data["losses"]["test"] = test_loss
    run_data["metrics"]["test"] = test_met
    run_data["predictions"] = y_pred_test
    run_data["ground_truth"] = y_true_test
    experiment_data["frozen_embeddings"]["SPR_BENCH"]["num_epochs"][
        f"epochs_{num_epochs}"
    ] = run_data
    print(f"Test HWA={test_met[2]:.4f}")

# -------------------- save + plot ---------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to {working_dir}/experiment_data.npy")

fig, ax = plt.subplots()
for k, v in experiment_data["frozen_embeddings"]["SPR_BENCH"]["num_epochs"].items():
    ax.plot(v["losses"]["val"], label=k)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss")
ax.set_title("Frozen-Emb GRU Validation Loss")
ax.legend()
plt.savefig(os.path.join(working_dir, "spr_frozen_emb_loss.png"))
plt.close(fig)
print("Loss curve saved.")
