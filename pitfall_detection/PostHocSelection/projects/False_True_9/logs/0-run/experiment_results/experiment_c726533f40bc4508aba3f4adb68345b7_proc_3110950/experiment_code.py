import os, pathlib, random, numpy as np, torch, torch.nn as nn, math
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility -----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ---------------- paths / device -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def find_spr_bench():
    cand = [os.environ.get("SPR_DATA_PATH")] if os.environ.get("SPR_DATA_PATH") else []
    cand += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).joinpath("train.csv").exists():
            return pathlib.Path(p).resolve()
    raise FileNotFoundError("SPR_BENCH not found")


DATA_PATH = find_spr_bench()
print("Found SPR_BENCH at:", DATA_PATH)


# ---------------- metrics helpers -----------------
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def swa(seq, y, yhat):
    w = [count_shape_variety(s) for s in seq]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0.0


def cwa(seq, y, yhat):
    w = [count_color_variety(s) for s in seq]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0.0


def hwa(s, c):
    return 2 * s * c / (s + c) if s + c else 0.0


# ---------------- dataset -------------------------
def load_spr(path):
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(path / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr(DATA_PATH)

# vocab
tok2id = {
    t: i + 1
    for i, t in enumerate(
        sorted({tok for ex in spr["train"] for tok in ex["sequence"].split()})
    )
}
PAD_ID = 0
vocab_size = len(tok2id) + 1
encode = lambda s: [tok2id[t] for t in s.split()]
num_classes = len(set(spr["train"]["label"]))
print("Vocab size:", vocab_size, "#classes:", num_classes)


class SPRTorchSet(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.enc = [encode(s) for s in self.seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.enc[i]),
            "label": torch.tensor(self.labels[i]),
            "raw_seq": self.seqs[i],
        }


def collate(b):
    m = max(len(x["input_ids"]) for x in b)
    ids = [
        torch.cat([x["input_ids"], torch.full((m - len(x["input_ids"]),), PAD_ID)])
        for x in b
    ]
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack([x["label"] for x in b]),
        "raw_seq": [x["raw_seq"] for x in b],
    }


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------- model ---------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, classes, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        emb = self.embed(x)
        lens = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        out = torch.cat([h[-2], h[-1]], 1)
        return self.fc(out)


# -------------- train / eval ----------------------
def run_experiment(depth, epochs=6, base_hidden=256):
    per_layer = max(1, base_hidden // depth)
    model = BiLSTMClassifier(vocab_size, 64, per_layer, num_classes, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0
        nb = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            out = model(batch["input_ids"])
            loss = crit(out, batch["label"])
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
        store["losses"]["train"].append((ep, tl / nb))
        model.eval()
        vl = 0
        nb = 0
        preds = []
        gts = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                out = model(batch["input_ids"])
                loss = crit(out, batch["label"])
                vl += loss.item()
                nb += 1
                p = out.argmax(-1).cpu().tolist()
                preds += p
                gts += batch["label"].cpu().tolist()
                seqs += batch["raw_seq"]
        vloss = vl / nb
        S = swa(seqs, gts, preds)
        C = cwa(seqs, gts, preds)
        H = hwa(S, C)
        store["losses"]["val"].append((ep, vloss))
        store["metrics"]["val"].append((ep, S, C, H))
        if ep == epochs:
            store["predictions"] = preds
            store["ground_truth"] = gts
        print(
            f"[depth={depth}] Epoch{ep} train_loss={tl/nb:.4f} val_loss={vloss:.4f} SWA={S:.4f} CWA={C:.4f} HWA={H:.4f}"
        )
    return store


# -------------- main sweep & save -----------------
experiment_data = {"layer_depth": {}}
for d in [1, 2, 3]:
    experiment_data["layer_depth"][d] = {"SPR_BENCH": run_experiment(d)}

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
