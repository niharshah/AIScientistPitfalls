import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility -----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ---------------- work dir & device ---------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- locate SPR_BENCH ---------------
def find_spr_bench() -> pathlib.Path:
    cand, env = [], os.environ.get("SPR_DATA_PATH")
    if env:
        cand.append(env)
    cand += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        fp = pathlib.Path(p).expanduser()
        if fp.joinpath("train.csv").exists():
            return fp.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------------- metric helpers -----------------
def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.split()))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ---------------- load dataset -------------------
def load_spr_bench(root):
    ld = lambda csv: load_dataset(
        "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(train=ld("train.csv"), dev=ld("dev.csv"), test=ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# ---------------- vocabulary ----------------------
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


def encode(seq):
    return [token2id[t] for t in seq.split()]


# ---------------- torch dataset ------------------
class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.enc[i], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "raw_seq": self.seqs[i],
        }


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, raw = [], [], []
    for b in batch:
        seq = b["input_ids"]
        if pad := maxlen - len(seq):
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
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


# ---------------- uni-directional model ----------
class UniLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = h_n[-1]
        return self.fc(out)


# ---------------- experiment container -----------
experiment_data = {"unidirectional_LSTM": {"SPR_BENCH": {"hidden_size": {}}}}


# ---------------- training loop ------------------
def run_experiment(hidden_size, epochs=6):
    model = UniLSTMClassifier(
        vocab_size, emb_dim=64, hidden=hidden_size, num_cls=num_classes
    ).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot, nb = 0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["label"])
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        store["losses"]["train"].append((epoch, tot / nb))
        # ---- validate ----
        model.eval()
        vtot, nb = 0, 0
        preds, labels, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = crit(logits, batch["label"])
                vtot += loss.item()
                nb += 1
                p = logits.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                labels.extend(l)
                seqs.extend(batch["raw_seq"])
        v_loss = vtot / nb
        store["losses"]["val"].append((epoch, v_loss))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["metrics"]["val"].append((epoch, swa, cwa, hwa))
        if epoch == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
        print(
            f"[hidden={hidden_size}] Epoch{epoch} "
            f"train_loss={store['losses']['train'][-1][1]:.4f} "
            f"val_loss={v_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )
    return store


# ---------------- hyper-parameter sweep ----------
for hs in [64, 128, 256, 512]:
    experiment_data["unidirectional_LSTM"]["SPR_BENCH"]["hidden_size"][hs] = (
        run_experiment(hs)
    )

# ---------------- save results -------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
