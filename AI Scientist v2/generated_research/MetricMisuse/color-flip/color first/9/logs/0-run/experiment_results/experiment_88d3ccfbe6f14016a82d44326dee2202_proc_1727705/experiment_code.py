import os, pathlib, random, time, math, numpy as np, torch
from typing import List
from collections import Counter
from datasets import load_dataset, DatasetDict
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------- misc / reproducibility --------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- load SPR_BENCH --------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Loaded splits:", {k: len(v) for k, v in spr.items()})


# -------------------- helpers --------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum((wt if yt == yp else 0) for wt, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wt if yt == yp else 0) for wt, yt, yp in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def dwhs(cwa, swa):
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) else 0.0


# -------------------- glyph clustering --------------------
def token_feature(tok: str) -> List[float]:
    chars = [ord(c) for c in tok]
    first = chars[0]
    rest_mean = sum(chars[1:]) / len(chars[1:]) if len(chars) > 1 else 0.0
    return [first, rest_mean]


all_tokens = sorted(
    {t for seq in spr["train"]["sequence"] for t in seq.strip().split()}
)
X = np.array([token_feature(t) for t in all_tokens])
k = max(8, int(math.sqrt(len(all_tokens))))
print(f"Clustering {len(all_tokens)} glyphs into {k} clusters…")
clusters = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
glyph2cluster = {tok: int(c) for tok, c in zip(all_tokens, clusters)}
print("Done clustering.")


# -------------------- datasets --------------------
class SPRClusteredDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [glyph2cluster.get(t, 0) + 1 for t in self.seqs[idx].strip().split()]
        return dict(
            input=torch.tensor(ids, dtype=torch.long),
            label=torch.tensor(int(self.labels[idx]), dtype=torch.long),
            raw_seq=self.seqs[idx],
        )


def collate_fn(batch):
    lens = [len(x["input"]) for x in batch]
    maxlen = max(lens)
    pad_inputs = [
        torch.cat([b["input"], torch.zeros(maxlen - len(b["input"]), dtype=torch.long)])
        for b in batch
    ]
    return dict(
        input=torch.stack(pad_inputs).to(device),
        label=torch.stack([b["label"] for b in batch]).to(device),
        len=torch.tensor(lens, dtype=torch.long).to(device),
        raw_seq=[b["raw_seq"] for b in batch],
    )


train_ds, dev_ds, test_ds = (
    SPRClusteredDataset(spr["train"]),
    SPRClusteredDataset(spr["dev"]),
    SPRClusteredDataset(spr["test"]),
)
train_loader = DataLoader(train_ds, 256, True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, 512, False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, 512, False, collate_fn=collate_fn)

num_labels = len(set(spr["train"]["label"]))
vocab_size = k + 2  # clusters + PAD + OOV
print(f"num_labels={num_labels}, vocab_size={vocab_size}")


# -------------------- model w/ dropout --------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, num_classes, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.emb_dp = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)  # single layer
        self.pre_fc_dp = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths):
        emb = self.emb_dp(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        out = self.pre_fc_dp(h.squeeze(0))
        return self.fc(out)


# -------------------- experiment container --------------------
experiment_data = {"dropout_rate": {}}


# -------------------- training / evaluation loop --------------------
def run_experiment(drop_rate: float, epochs=5):
    tag = f"{drop_rate:.2f}"
    experiment_data["dropout_rate"][tag] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = GRUClassifier(vocab_size, 32, 64, num_labels, drop_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # training epochs
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input"], batch["len"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch["label"].size(0)
        tr_loss = ep_loss / len(train_ds)
        experiment_data["dropout_rate"][tag]["losses"]["train"].append((ep, tr_loss))
        # ---- validation ----
        model.eval()
        v_loss = 0.0
        preds = labels = seqs = []
        all_preds, all_labels, all_seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = model(batch["input"], batch["len"])
                loss = criterion(logits, batch["label"])
                v_loss += loss.item() * batch["label"].size(0)
                pred = logits.argmax(-1).cpu().tolist()
                lab = batch["label"].cpu().tolist()
                all_preds.extend(pred)
                all_labels.extend(lab)
                all_seqs.extend(batch["raw_seq"])
        v_loss /= len(dev_ds)
        cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
        swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
        val_dwhs = dwhs(cwa, swa)
        experiment_data["dropout_rate"][tag]["losses"]["val"].append((ep, v_loss))
        experiment_data["dropout_rate"][tag]["metrics"]["val"].append(
            (ep, cwa, swa, val_dwhs)
        )
        print(
            f"[dropout={drop_rate}] Epoch {ep}: val_loss={v_loss:.4f} CWA={cwa:.3f} SWA={swa:.3f} DWHS={val_dwhs:.3f}"
        )
    # ---------------- test evaluation ----------------
    model.eval()
    t_preds, t_labels, t_seqs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input"], batch["len"])
            t_preds.extend(logits.argmax(-1).cpu().tolist())
            t_labels.extend(batch["label"].cpu().tolist())
            t_seqs.extend(batch["raw_seq"])
    cwa = color_weighted_accuracy(t_seqs, t_labels, t_preds)
    swa = shape_weighted_accuracy(t_seqs, t_labels, t_preds)
    test_dwhs = dwhs(cwa, swa)
    print(
        f"[dropout={drop_rate}] TEST  CWA={cwa:.3f} SWA={swa:.3f} DWHS={test_dwhs:.3f}"
    )
    ed = experiment_data["dropout_rate"][tag]
    ed["predictions"], ed["ground_truth"] = t_preds, t_labels
    ed["metrics"]["test"] = (cwa, swa, test_dwhs)


# -------------------- hyper-parameter sweep --------------------
for rate in [0.0, 0.2, 0.3, 0.4, 0.5]:
    run_experiment(rate)

# -------------------- save all --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
