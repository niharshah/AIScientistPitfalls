# ------------------- set-up & imports -------------------
import os, pathlib, csv, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ------------------- helper : build dummy corpus if missing -------------------
def _generate_dummy_csv(path: pathlib.Path, n_rows: int):
    shapes = ["A", "B", "C", "D"]
    colors = ["1", "2", "3"]
    with path.open("w", newline="") as f:
        wrt = csv.writer(f)
        wrt.writerow(["id", "sequence", "label"])
        for i in range(n_rows):
            length = random.randint(4, 8)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(length)
            )
            # toy rule: label 1 if #unique shapes > #unique colors else 0
            y = int(len({s[0] for s in seq.split()}) > len({s[1] for s in seq.split()}))
            wrt.writerow([i, seq, y])


def find_or_build_spr_bench() -> pathlib.Path:
    candidates = []
    if os.environ.get("SPR_DATA_PATH"):
        candidates.append(os.environ["SPR_DATA_PATH"])
    candidates += ["./SPR_BENCH", "../SPR_BENCH", "../../SPR_BENCH"]
    for p in candidates:
        pth = pathlib.Path(p).expanduser()
        if pth.joinpath("train.csv").exists():
            print(f"Found existing SPR_BENCH at {pth.resolve()}")
            return pth.resolve()

    # not found – create tiny synthetic one
    synth_root = pathlib.Path(working_dir) / "SPR_BENCH"
    synth_root.mkdir(parents=True, exist_ok=True)
    print(f"Creating synthetic SPR_BENCH at {synth_root.resolve()} for demo purposes.")
    _generate_dummy_csv(synth_root / "train.csv", 500)
    _generate_dummy_csv(synth_root / "dev.csv", 100)
    _generate_dummy_csv(synth_root / "test.csv", 100)
    return synth_root.resolve()


DATA_PATH = find_or_build_spr_bench()


# ------------------- metrics -------------------
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


def difficulty_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ------------------- load dataset -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# ------------------- vocab -------------------
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
UNK_ID = len(token2id) + 1
token2id["<UNK>"] = UNK_ID
vocab_size = len(token2id) + 1  # +1 for padding index 0


def encode(seq):
    return [token2id.get(tok, UNK_ID) for tok in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ------------------- torch dataset -------------------
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
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, raw = [], [], []
    for b in batch:
        seq = b["input_ids"]
        pad_len = maxlen - len(seq)
        if pad_len:
            seq = torch.cat([seq, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        ids.append(seq)
        labels.append(b["label"])
        raw.append(b["raw_seq"])
    return {"input_ids": torch.stack(ids), "label": torch.stack(labels), "raw_seq": raw}


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ------------------- model -------------------
class UniLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


# ------------------- experiment container -------------------
experiment_data = {"Unidirectional_LSTM": {}}


# ------------------- training loop -------------------
def run_experiment(hidden_size, epochs=6):
    model = UniLSTMClassifier(vocab_size, 64, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot_loss = 0
        n_batches = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            n_batches += 1
        avg_train = tot_loss / n_batches
        store["losses"]["train"].append((ep, avg_train))

        # ---- validation ----
        model.eval()
        v_loss = 0
        v_batches = 0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item()
                v_batches += 1
                p = logits.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                gts.extend(l)
                seqs.extend(batch["raw_seq"])
        avg_val = v_loss / v_batches
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        dwa = difficulty_weighted_accuracy(seqs, gts, preds)
        store["losses"]["val"].append((ep, avg_val))
        store["metrics"]["val"].append((ep, swa, cwa, hwa, dwa))
        if ep == epochs:
            store["predictions"] = preds
            store["ground_truth"] = gts
        print(
            f"[hidden={hidden_size}] Epoch{ep} train_loss={avg_train:.4f} "
            f"val_loss={avg_val:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f} DWA={dwa:.4f}"
        )
    return store


# ------------------- hyperparameter sweep -------------------
for hs in [64, 128, 256, 512]:
    experiment_data["Unidirectional_LSTM"][hs] = {
        "SPR_BENCH": run_experiment(hs, epochs=4)
    }

# ------------------- save experiment data -------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f'Saved experiment data to {os.path.join(working_dir, "experiment_data.npy")}')
