# Aggregation-Head Ablation : Final-State vs Mean-Pooling
import os, pathlib, numpy as np, torch, torch.nn as nn, random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- reproducibility ----------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ---------- work dir & device --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def find_spr_bench() -> pathlib.Path:
    cand = [os.environ.get("SPR_DATA_PATH")] if os.environ.get("SPR_DATA_PATH") else []
    cand += [
        "./SPR_BENCH",
        "../SPR_BENCH",
        "../../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for p in cand:
        if p and pathlib.Path(p).expanduser().joinpath("train.csv").exists():
            return pathlib.Path(p).expanduser().resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found")


DATA_PATH = find_spr_bench()
print(f"Found SPR_BENCH at: {DATA_PATH}")


# ---------- metrics ----------
def count_shape_variety(s):
    return len(set(tok[0] for tok in s.strip().split() if tok))


def count_color_variety(s):
    return len(set(tok[1] for tok in s.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0


# ---------- dataset ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    _ld = lambda csv: load_dataset(
        "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
    )
    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)

# ---------- vocab ----------
all_tokens = set(tok for ex in spr["train"] for tok in ex["sequence"].split())
token2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
PAD_ID = 0
vocab_size = len(token2id) + 1


def encode(seq: str):
    return [token2id[t] for t in seq.split()]


num_classes = len(set(spr["train"]["label"]))
print(f"Vocab size={vocab_size}, num_classes={num_classes}")


# ---------- torch dataset ----------
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
        pad = maxlen - len(seq)
        if pad:
            seq = torch.cat([seq, torch.full((pad,), PAD_ID, dtype=torch.long)])
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


# ---------- model variants ----------
class BiLSTM_FinalState(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        out = torch.cat([h_n[-2], h_n[-1]], 1)
        return self.fc(out)


class BiLSTM_MeanPool(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden, num_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_cls)

    def forward(self, x):
        emb = self.embed(x)
        lengths = (x != PAD_ID).sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        mask = (x != PAD_ID).unsqueeze(-1).to(out.dtype)
        summed = (out * mask).sum(1)
        mean = summed / lengths.unsqueeze(-1).clamp(min=1)
        return self.fc(mean)


# ---------- experiment container ----------
experiment_data = {
    "aggregation_head_final_state": {"SPR_BENCH": {"hidden_size": {}}},
    "aggregation_head_mean_pool": {"SPR_BENCH": {"hidden_size": {}}},
}


# ---------- training routine ----------
def train_eval(model, epochs=6):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tloss = 0
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
            tloss += loss.item()
            nb += 1
        store["losses"]["train"].append((epoch, tloss / nb))
        # validate
        model.eval()
        vloss = 0
        nb = 0
        preds, labels, seqs = [], [], []
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
                p = logit.argmax(-1).cpu().tolist()
                l = batch["label"].cpu().tolist()
                preds.extend(p)
                labels.extend(l)
                seqs.extend(batch["raw_seq"])
        vloss /= nb
        swa = shape_weighted_accuracy(seqs, labels, preds)
        cwa = color_weighted_accuracy(seqs, labels, preds)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        store["losses"]["val"].append((epoch, vloss))
        store["metrics"]["val"].append((epoch, swa, cwa, hwa))
        if epoch == epochs:
            store["predictions"] = preds
            store["ground_truth"] = labels
        print(
            f"Epoch{epoch} train_loss={tloss/nb:.4f} val_loss={vloss:.4f} "
            f"SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}"
        )
    return store


# ---------- run experiments ----------
hidden_sizes = [64, 128, 256, 512]
for hs in hidden_sizes:
    print(f"\n=== Final-State head | hidden={hs} ===")
    model = BiLSTM_FinalState(vocab_size, 64, hs, num_classes)
    res = train_eval(model)
    experiment_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][
        hs
    ] = res

for hs in hidden_sizes:
    print(f"\n=== Mean-Pool head | hidden={hs} ===")
    model = BiLSTM_MeanPool(vocab_size, 64, hs, num_classes)
    res = train_eval(model)
    experiment_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][hs] = res

# ---------- save ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
