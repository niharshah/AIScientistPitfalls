# bi_lstm_ablation.py
import os, pathlib, math, time, random, string, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- dirs / device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load SPR_BENCH or synth ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH missing: synthesising toy data")

    def synth(n):
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:12], k=random.randint(5, 15))
            )
            yield {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}

    spr = DatasetDict(
        {
            "train": load_dataset(
                "json", data_files={"train": list(synth(4000))}, split="train"
            ),
            "dev": load_dataset(
                "json", data_files={"train": list(synth(800))}, split="train"
            ),
            "test": load_dataset(
                "json", data_files={"train": list(synth(800))}, split="train"
            ),
        }
    )
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vsize = len(vocab)
MAX_LEN = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)


def enc(seq):
    ids = [vocab["<cls>"]] + [vocab.get(c, 1) for c in seq][: MAX_LEN - 1]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


def complexity(ex):
    return float(len(set(ex["sequence"])))


class SPRTorch(Dataset):
    def __init__(self, hf):
        self.d = hf

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        ex = self.d[idx]
        return {
            "input_ids": torch.tensor(enc(ex["sequence"]), dtype=torch.long),
            "labels": torch.tensor(int(ex["label"]), dtype=torch.long),
            "weights": torch.tensor(
                float(ex.get("complexity", complexity(ex))), dtype=torch.float
            ),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_ds, dev_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)


# ---------- Bi-LSTM backbone ----------
class CharBiLSTM(nn.Module):
    def __init__(self, v, d_model=128, num_cls=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(v, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        hidden = d_model // 2  # bidirectional => hidden*2 = d_model
        self.lstm = nn.LSTM(
            d_model,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_cls)

    def forward(self, x):
        mask = x == 0
        lens = (~mask).sum(1).clamp(min=1).cpu()
        h = self.emb(x) + self.pos[:, : x.size(1)]
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lens, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (2,batch,hidden)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, d_model)
        h_cat = self.norm(h_cat)
        return self.fc(h_cat)


# ---------- utils ----------
def cwa(pred, lab, w):
    correct = (pred == lab).astype(float)
    return (correct * w).sum() / w.sum()


def curriculum(epoch, total):
    return min(1.0, (epoch + 1) / (total / 2))


# ---------- training settings ----------
batch = 32
epochs = 12
model = CharBiLSTM(vsize).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

# ---------- experiment data ----------
experiment_data = {
    "bi_lstm_backbone": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "weights": [],
        }
    }
}

best_val, bad_epochs = 1e9, 0
for epoch in range(epochs):
    model.train()
    cur_w = curriculum(epoch, epochs)
    tot_loss, items = 0, 0
    for batch_d in train_loader:
        batch_d = {k: v.to(device) for k, v in batch_d.items()}
        opt.zero_grad()
        logits = model(batch_d["input_ids"])
        loss = criterion(logits, batch_d["labels"])
        loss = (loss * torch.where(batch_d["weights"] > 5, cur_w, 1.0)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item() * batch_d["labels"].size(0)
        items += batch_d["labels"].size(0)
    train_loss = tot_loss / items
    experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )

    # ---- validation ----
    model.eval()
    vloss, vitems = 0, 0
    preds, labels, weights = [], [], []
    with torch.no_grad():
        for batch_d in dev_loader:
            batch_d = {k: v.to(device) for k, v in batch_d.items()}
            out = model(batch_d["input_ids"])
            loss = criterion(out, batch_d["labels"])
            vloss += loss.item() * batch_d["labels"].size(0)
            vitems += batch_d["labels"].size(0)
            p = out.argmax(1).cpu().numpy()
            l = batch_d["labels"].cpu().numpy()
            w = batch_d["weights"].cpu().numpy()
            preds.extend(p)
            labels.extend(l)
            weights.extend(w)
    vloss /= vitems
    mf1 = f1_score(labels, preds, average="macro")
    cw = cwa(np.array(preds), np.array(labels), np.array(weights))
    experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["losses"]["val"].append(vloss)
    experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": mf1, "cwa": cw}
    )
    print(f"Epoch {epoch+1}: val_loss={vloss:.4f} | Macro-F1={mf1:.3f} | CWA={cw:.3f}")
    if vloss < best_val - 1e-4:
        best_val = vloss
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= 3:
        print("Early stopping.")
        break
    sched.step()

experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["predictions"] = preds
experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["ground_truth"] = labels
experiment_data["bi_lstm_backbone"]["SPR_BENCH"]["weights"] = weights
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
