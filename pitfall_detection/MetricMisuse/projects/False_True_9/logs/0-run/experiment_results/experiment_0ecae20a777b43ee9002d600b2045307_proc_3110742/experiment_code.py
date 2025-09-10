import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility -----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ---------------- working dir / device -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench()
print("Found SPR_BENCH at:", DATA_PATH)


# ---------------- metrics helpers ----------------
def count_shape_variety(s):
    return len(set(t[0] for t in s.split() if t))


def count_color_variety(s):
    return len(set(t[1] for t in s.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if swa + cwa > 0 else 0.0


# ---------------- load dataset -------------------
def load_spr(root: pathlib.Path) -> DatasetDict:
    load = lambda csv: load_dataset(
        "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(
        train=load("train.csv"), dev=load("dev.csv"), test=load("test.csv")
    )


spr = load_spr(DATA_PATH)

# ---------------- vocabulary ---------------------
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
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


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, lab, raw = [], [], []
    for b in batch:
        seq = b["input_ids"]
        if pad := maxlen - len(seq):
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
        ids.append(seq)
        lab.append(b["label"])
        raw.append(b["raw_seq"])
    return {"input_ids": torch.stack(ids), "label": torch.stack(lab), "raw_seq": raw}


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ---------------- bag-of-embeddings model --------
class BagOfEmbeddingsClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, x):
        emb = self.embed(x)  # (B,L,E)
        mask = (x != PAD_ID).unsqueeze(-1)  # (B,L,1)
        summed = (emb * mask).sum(1)  # (B,E)
        lengths = mask.sum(1).clamp(min=1)  # (B,1)
        avg = summed / lengths  # (B,E)
        return self.fc(avg)


# ---------------- experiment container ----------
experiment_data = {
    "bag_of_embeddings": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# ---------------- training loop -----------------
def run_boe(epochs=6, emb_dim=64):
    model = BagOfEmbeddingsClassifier(vocab_size, emb_dim, num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = experiment_data["bag_of_embeddings"]["SPR_BENCH"]
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tloss = nb = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logit = model(batch["input_ids"])
            loss = crit(logit, batch["label"])
            loss.backward()
            opt.step()
            tloss += loss.item()
            nb += 1
        tr_loss = tloss / nb
        store["losses"]["train"].append((ep, tr_loss))
        # ---- validate ----
        model.eval()
        vloss = nb = 0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logit = model(batch["input_ids"])
                loss = crit(logit, batch["label"])
                vloss += loss.item()
                nb += 1
                preds.extend(logit.argmax(-1).cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
                seqs.extend(batch["raw_seq"])
        v_loss = vloss / nb
        store["losses"]["val"].append((ep, v_loss))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["metrics"]["val"].append((ep, swa, cwa, hwa))
        print(
            f"[BagEmb] Epoch{ep} train_loss={tr_loss:.4f} "
            f"val_loss={v_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )
        if ep == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
    return store


run_boe()

# ---------------- save results -------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
