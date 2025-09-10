import os, pathlib, numpy as np, torch, torch.nn as nn, random, math, time
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict


# ----------------- helpers & seeds -----------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------- working dir ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def find_spr_bench() -> pathlib.Path:
    cand = [os.environ.get("SPR_DATA_PATH")] + [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ----------------- dataset utils -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


spr = load_spr_bench(DATA_PATH)

# vocab
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# torch dataset
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return dict(
            input_ids=torch.tensor(self.enc[i], dtype=torch.long),
            label=torch.tensor(self.labels[i], dtype=torch.long),
            raw_seq=self.seqs[i],
        )


def collate_fn(batch):
    maxlen = max(len(it["input_ids"]) for it in batch)
    ids, labels, raw = [], [], []
    for it in batch:
        seq = it["input_ids"]
        if len(seq) < maxlen:
            seq = torch.cat(
                [seq, torch.full((maxlen - len(seq),), PAD_ID, dtype=torch.long)]
            )
        ids.append(seq)
        labels.append(it["label"])
        raw.append(it["raw_seq"])
    return dict(input_ids=torch.stack(ids), label=torch.stack(labels), raw_seq=raw)


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128, num_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.emb(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.fc(out)


# ----------------- experiment container -----------
CLIP_VALS = [0, 0.5, 1, 2, 5]
experiment_data = {"gradient_clipping_max_norm": {}}

EPOCHS = 6
for clip_idx, clip_val in enumerate(CLIP_VALS):
    print(f"\n=== Training with clip_val={clip_val} ===")
    set_seed(42 + clip_idx)
    model = BiLSTMClassifier(vocab_size, num_cls=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exp_cur = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        total_loss = 0
        n_batch = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        train_loss = total_loss / n_batch
        exp_cur["losses"]["train"].append((epoch, train_loss))

        # --- val ---
        model.eval()
        val_loss_tot = 0
        nb = 0
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                val_loss_tot += loss.item()
                nb += 1
                preds = logits.argmax(-1).cpu().tolist()
                labels = batch["label"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_seqs.extend(batch["raw_seq"])
        val_loss = val_loss_tot / nb
        exp_cur["losses"]["val"].append((epoch, val_loss))

        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        exp_cur["metrics"]["val"].append((epoch, swa, cwa, hwa))
        exp_cur["predictions"] = all_preds
        exp_cur["ground_truth"] = all_labels

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )

    experiment_data["gradient_clipping_max_norm"][str(clip_val)] = exp_cur

# ---------------- save all ------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
