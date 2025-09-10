import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------ #
#                      reproducibility & folders                      #
# ------------------------------------------------------------------ #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------ #
#                        locate SPR_BENCH                            #
# ------------------------------------------------------------------ #
def find_spr_bench() -> pathlib.Path:
    cands = []
    if p := os.environ.get("SPR_DATA_PATH"):
        cands.append(p)
    cands += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cands:
        pth = pathlib.Path(p).expanduser()
        if pth.joinpath("train.csv").exists():
            return pth.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ------------------------------------------------------------------ #
#                 metric helpers (unchanged from base)               #
# ------------------------------------------------------------------ #
def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.split()))


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ------------------------------------------------------------------ #
#                       load original dataset                        #
# ------------------------------------------------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)

# ------------------------------------------------------------------ #
#                 build shared vocabulary from SPR_BENCH             #
# ------------------------------------------------------------------ #
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1
encode = lambda seq: [token2id[t] for t in seq.split()]
num_classes_spr = len(set(spr["train"]["label"]))
print(f"Vocab={vocab_size}, SPR num_classes={num_classes_spr}")

# ------------------------------------------------------------------ #
#                     extract shape / color sets                     #
# ------------------------------------------------------------------ #
shape_set = sorted({tok[0] for tok in all_tokens})
color_set = sorted({tok[1] for tok in all_tokens})
shape2id = {s: i for i, s in enumerate(shape_set)}
color2id = {c: i for i, c in enumerate(color_set)}
print(f"Shapes:{shape_set}, Colors:{color_set}")


# ------------------------------------------------------------------ #
#                    synthetic dataset generation                    #
# ------------------------------------------------------------------ #
def generate_sequence(target_major, major_type: str, length_rng=(4, 8)):
    """major_type: 'shape' or 'color'"""
    n = random.randint(*length_rng)
    seq_tokens = []
    tokens_by_shape = {s: [t for t in all_tokens if t[0] == s] for s in shape_set}
    tokens_by_color = {c: [t for t in all_tokens if t[1] == c] for c in color_set}
    if major_type == "shape":
        maj_shape = target_major
        maj_cnt = random.randint(n // 2 + 1, n)
        # majority tokens
        for _ in range(maj_cnt):
            seq_tokens.append(random.choice(tokens_by_shape[maj_shape]))
        # fill rest
        others = [s for s in shape_set if s != maj_shape]
        for _ in range(n - maj_cnt):
            sh = random.choice(others)
            seq_tokens.append(random.choice(tokens_by_shape[sh]))
    else:  # color majority
        maj_color = target_major
        maj_cnt = random.randint(n // 2 + 1, n)
        for _ in range(maj_cnt):
            seq_tokens.append(random.choice(tokens_by_color[maj_color]))
        others = [c for c in color_set if c != maj_color]
        for _ in range(n - maj_cnt):
            col = random.choice(others)
            seq_tokens.append(random.choice(tokens_by_color[col]))
    random.shuffle(seq_tokens)
    return " ".join(seq_tokens)


def build_synthetic_dataset(kind: str, size: int):
    # kind: 'shape' or 'color'
    seqs, labels = [], []
    for _ in range(size):
        if kind == "shape":
            tar = random.choice(shape_set)
            seqs.append(generate_sequence(tar, "shape"))
            labels.append(shape2id[tar])
        else:
            tar = random.choice(color_set)
            seqs.append(generate_sequence(tar, "color"))
            labels.append(color2id[tar])
    return {"sequence": seqs, "label": labels}


syn_shape = build_synthetic_dataset("shape", size=len(spr["train"]))
syn_color = build_synthetic_dataset("color", size=len(spr["train"]))


# ------------------------------------------------------------------ #
#                     torch Dataset wrappers                         #
# ------------------------------------------------------------------ #
class TorchSeqSet(Dataset):
    """General wrapper for any sequence-label set."""

    def __init__(self, seqs, labels, task: str):
        self.seqs = seqs
        self.labels = labels
        self.task = task
        self.enc = [encode(s) for s in seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
            "task": self.task,
        }


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, raw, task = [], [], [], []
    for b in batch:
        seq = b["input_ids"]
        if pad := maxlen - len(seq):
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
        ids.append(seq)
        labels.append(b["label"])
        raw.append(b["raw_seq"])
        task.append(b["task"])
    return {
        "input_ids": torch.stack(ids),
        "label": torch.stack(labels),
        "raw_seq": raw,
        "task": task,
    }


# SPR loaders
spr_train_loader = DataLoader(
    TorchSeqSet(spr["train"]["sequence"], spr["train"]["label"], task="SPR"),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)
spr_dev_loader = DataLoader(
    TorchSeqSet(spr["dev"]["sequence"], spr["dev"]["label"], task="SPR"),
    batch_size=256,
    shuffle=False,
    collate_fn=collate,
)

# Synthetic loaders (train only)
shape_train_loader = DataLoader(
    TorchSeqSet(syn_shape["sequence"], syn_shape["label"], task="SHAPE"),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)
color_train_loader = DataLoader(
    TorchSeqSet(syn_color["sequence"], syn_color["label"], task="COLOR"),
    batch_size=128,
    shuffle=True,
    collate_fn=collate,
)


# ------------------------------------------------------------------ #
#                       multi-task model                             #
# ------------------------------------------------------------------ #
class MultiTaskBiLSTM(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, n_cls_spr, n_cls_shape, n_cls_color):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        d = hidden * 2
        self.heads = nn.ModuleDict(
            {
                "SPR": nn.Linear(d, n_cls_spr),
                "SHAPE": nn.Linear(d, n_cls_shape),
                "COLOR": nn.Linear(d, n_cls_color),
            }
        )

    def forward(self, x, task):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        feats = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.heads[task](feats)


# ------------------------------------------------------------------ #
#                       training helpers                             #
# ------------------------------------------------------------------ #
def run_baseline(hidden, epochs=6):
    model = MultiTaskBiLSTM(
        vocab_size, 64, hidden, num_classes_spr, len(shape_set), len(color_set)
    ).to(device)
    criterion = {"SPR": nn.CrossEntropyLoss()}
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = {
        "metrics": {"val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        # train on SPR only
        model.train()
        tl = 0
        nb = 0
        for batch in spr_train_loader:
            optim.zero_grad()
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            logits = model(x, "SPR")
            loss = criterion["SPR"](logits, y)
            loss.backward()
            optim.step()
            tl += loss.item()
            nb += 1
        store["losses"]["train"].append((ep, tl / nb))
        # validate
        model.eval()
        vl = 0
        nb = 0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in spr_dev_loader:
                x = batch["input_ids"].to(device)
                y = batch["label"].to(device)
                lo = model(x, "SPR")
                loss = criterion["SPR"](lo, y)
                vl += loss.item()
                nb += 1
                preds.extend(lo.argmax(-1).cpu().tolist())
                labels.extend(y.cpu().tolist())
                seqs.extend(batch["raw_seq"])
        vl /= nb
        store["losses"]["val"].append((ep, vl))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["metrics"]["val"].append((ep, swa, cwa, hwa))
        if ep == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
        print(f"[BL hidden={hidden}]Ep{ep} loss={tl/nb:.4f}/{vl:.4f} HWA={hwa:.4f}")
    return store


def combined_train_loader():
    """yield batches from the three loaders in round-robin fashion"""
    it_spr = iter(spr_train_loader)
    it_shape = iter(shape_train_loader)
    it_color = iter(color_train_loader)
    while True:
        try:
            yield next(it_spr)
        except StopIteration:
            it_spr = iter(spr_train_loader)
            yield next(it_spr)
        try:
            yield next(it_shape)
        except StopIteration:
            it_shape = iter(shape_train_loader)
            yield next(it_shape)
        try:
            yield next(it_color)
        except StopIteration:
            it_color = iter(color_train_loader)
            yield next(it_color)


def run_mixed(hidden, epochs=6, steps_per_epoch=len(spr_train_loader)):
    model = MultiTaskBiLSTM(
        vocab_size, 64, hidden, num_classes_spr, len(shape_set), len(color_set)
    ).to(device)
    # criteria
    crit = {
        "SPR": nn.CrossEntropyLoss(),
        "SHAPE": nn.CrossEntropyLoss(),
        "COLOR": nn.CrossEntropyLoss(),
    }
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = {
        "metrics": {"val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    loader = combined_train_loader()
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0
        for step in range(steps_per_epoch):
            batch = next(loader)
            opt.zero_grad()
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            task = batch["task"][0]
            logits = model(x, task)
            loss = crit[task](logits, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        store["losses"]["train"].append((ep, tl / steps_per_epoch))
        # validate on SPR dev
        model.eval()
        vl = 0
        nb = 0
        preds = []
        labels = []
        seqs = []
        with torch.no_grad():
            for batch in spr_dev_loader:
                x = batch["input_ids"].to(device)
                y = batch["label"].to(device)
                lo = model(x, "SPR")
                loss = crit["SPR"](lo, y)
                vl += loss.item()
                nb += 1
                preds.extend(lo.argmax(-1).cpu().tolist())
                labels.extend(y.cpu().tolist())
                seqs.extend(batch["raw_seq"])
        vl /= nb
        store["losses"]["val"].append((ep, vl))
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["metrics"]["val"].append((ep, swa, cwa, hwa))
        if ep == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
        print(
            f"[MIX hidden={hidden}]Ep{ep} loss={tl/steps_per_epoch:.4f}/{vl:.4f} HWA={hwa:.4f}"
        )
    return store


# ------------------------------------------------------------------ #
#                           experiments                              #
# ------------------------------------------------------------------ #
experiment_data = {"cross_dataset_generalization": {}}
hidden_list = [64, 128, 256, 512]
for hs in hidden_list:
    experiment_data["cross_dataset_generalization"][hs] = {
        "SPR_BENCH": run_baseline(hs),
        "SPR+SHAPE+COLOR": run_mixed(hs),
    }

# ------------------------------------------------------------------ #
#                               save                                 #
# ------------------------------------------------------------------ #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
