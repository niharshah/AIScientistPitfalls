import os, pathlib, numpy as np, torch, torch.nn as nn, random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- reproducibility -----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ---------------- working dir & device -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- locate SPR_BENCH ----------------
def find_spr_bench() -> pathlib.Path:
    cands, env = [], os.environ.get("SPR_DATA_PATH")
    if env:
        cands.append(env)
    cands += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cands:
        fp = pathlib.Path(p).expanduser()
        if fp.joinpath("train.csv").exists():
            return fp.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------------- metrics helpers ----------------
def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wt for wt, t, p in zip(w, y_t, y_p) if t == p) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if swa + cwa > 0 else 0.0


# ---------------- load dataset -------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# ---------------- vocabulary ----------------------
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


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, raw = [], [], []
    for it in batch:
        seq = it["input_ids"]
        if maxlen - len(seq):
            seq = torch.cat(
                [seq, torch.full((maxlen - len(seq),), PAD_ID, dtype=torch.long)]
            )
        ids.append(seq)
        labels.append(it["label"])
        raw.append(it["raw_seq"])
    return {"input_ids": torch.stack(ids), "label": torch.stack(labels), "raw_seq": raw}


train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ---------------- Bag-of-Words model -------------
class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, x):
        emb = self.embed(x)  # B,L,E
        mask = (x != PAD_ID).unsqueeze(-1)  # B,L,1
        summed = (emb * mask).sum(1)  # B,E
        lens = mask.sum(1).clamp(min=1)  # B,1
        mean = summed / lens  # B,E
        return self.fc(mean)


# ---------------- training procedure -------------
def run_bow_experiment(emb_dim=64, epochs=6):
    model = BagOfWordsClassifier(vocab_size, emb_dim, num_cls=num_classes).to(device)
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
        tot_loss = 0
        nb = 0
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
            tot_loss += loss.item()
            nb += 1
        store["losses"]["train"].append((epoch, tot_loss / nb))
        # ---- validate ----
        model.eval()
        vloss = 0
        nb = 0
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
        store["losses"]["val"].append((epoch, v_loss))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["metrics"]["val"].append((epoch, swa, cwa, hwa))
        if epoch == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
        print(
            f"[BagOfWords] Ep{epoch} train_loss={tot_loss/nb:.4f} "
            f"val_loss={v_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )
    return store


# ---------------- run & save ---------------------
experiment_data = {"bag_of_words": {"SPR_BENCH": run_bow_experiment()}}
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved experiment data to {os.path.join(working_dir,'experiment_data.npy')}")
