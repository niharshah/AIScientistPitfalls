import os, pathlib, random, csv, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ----------------- working dir -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- device ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- reproducibility ----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# -------------------------------------------------
def _make_dummy_spr(path: pathlib.Path, n_train=256, n_dev=64, n_test=64):
    """
    Create a minimal synthetic SPR_BENCH folder with three csv files
    so that the rest of the pipeline can run even when the real
    dataset is unavailable.
    """
    print("Creating dummy SPR_BENCH dataset …")
    shapes, colors = list("ABCDE"), list("01234")

    def rand_seq():
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 12))
        ]
        return " ".join(toks)

    def write_csv(fname, n_rows):
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for idx in range(n_rows):
                seq = rand_seq()
                label = random.randint(0, 3)
                w.writerow([idx, seq, label])

    for split, n_rows in zip(["train", "dev", "test"], [n_train, n_dev, n_test]):
        write_csv(path / f"{split}.csv", n_rows)


def find_or_create_spr_bench() -> pathlib.Path:
    """
    Search common paths for SPR_BENCH; otherwise build a dummy set
    under ./working/SPR_BENCH_DUMMY so that code always runs.
    """
    env_path = os.environ.get("SPR_DATA_PATH")
    candidates = [env_path] if env_path else []
    candidates += ["./SPR_BENCH", "../SPR_BENCH", "../../SPR_BENCH"]
    for p in candidates:
        if p:
            pth = pathlib.Path(p).expanduser()
            if pth.joinpath("train.csv").exists():  # real train.csv
                return pth.resolve()
            if pth.joinpath("SPR_BENCH/train.csv").exists():  # nested
                return pth.joinpath("SPR_BENCH").resolve()
    # not found → create dummy
    dummy_root = pathlib.Path(working_dir) / "SPR_BENCH_DUMMY"
    dummy_root.mkdir(parents=True, exist_ok=True)
    _make_dummy_spr(dummy_root)
    return dummy_root.resolve()


DATA_PATH = find_or_create_spr_bench()
print("Using SPR_BENCH at:", DATA_PATH)


# ---------------- metric helpers -----------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def difficulty_weight(sequence: str) -> int:
    return count_shape_variety(sequence) + count_color_variety(sequence)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def difficulty_weighted_accuracy(seqs, y_t, y_p):
    w = [difficulty_weight(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0


# ---------------- load dataset -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_file: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


spr = load_spr_bench(DATA_PATH)

# ---------------- vocabulary ----------------------
all_tokens = {tok for ex in spr["train"] for tok in ex["sequence"].split()}
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1
num_classes = len(set(spr["train"]["label"]))


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ---------------- torch dataset ------------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate_fn(batch):
    maxlen = max(len(item["input_ids"]) for item in batch)
    ids, labs, raws = [], [], []
    for it in batch:
        pad_len = maxlen - len(it["input_ids"])
        seq_ids = (
            torch.cat(
                [it["input_ids"], torch.full((pad_len,), PAD_ID, dtype=torch.long)]
            )
            if pad_len
            else it["input_ids"]
        )
        ids.append(seq_ids)
        labs.append(it["label"])
        raws.append(it["raw_seq"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack(labs),
        "raw_seq": raws,
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# --------------- model ---------------------------
class BiLSTMClassifierNoMask(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)  # (B,T,E)
        _, (h_n, _) = self.lstm(emb)
        rep = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(rep)


# --------------- experiment container ------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# --------------- training loop -------------------
def run_experiment(hidden_size, epochs=3):
    model = BiLSTMClassifierNoMask(vocab_size, 64, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(batch["input_ids"])
            loss = criterion(out, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, avg_train_loss))

        # ---- Validation ----
        model.eval()
        val_loss, preds, labels, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = model(batch["input_ids"])
                loss = criterion(out, batch["label"])
                val_loss += loss.item()
                preds.extend(out.argmax(-1).cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        avg_val_loss = val_loss / len(dev_loader)
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        dwa = difficulty_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, avg_val_loss))
        experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, swa, cwa, dwa, hwa))
        print(
            f"[hid={hidden_size}] Ep{ep} TL={avg_train_loss:.4f} VL={avg_val_loss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} DWA={dwa:.4f} HWA={hwa:.4f}"
        )
    # store last predictions/labels
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = labels


# --------------- run a small sweep ---------------
for hs in [64, 128]:
    run_experiment(hs, epochs=3)

# --------------- save results --------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
