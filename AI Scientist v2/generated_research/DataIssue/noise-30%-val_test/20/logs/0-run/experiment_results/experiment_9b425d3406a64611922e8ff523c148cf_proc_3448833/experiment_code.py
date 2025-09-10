import os, pathlib, math, time, random, string, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------- paths / device --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- data loading ---------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    out["train"], out["dev"], out["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return out


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("SPR_BENCH missing – building tiny synthetic set")

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

# --------------- vocab / encoding ------------
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


def complexity(ex):  # proxy complexity
    return float(len(set(ex["sequence"])))


class SPRTorch(Dataset):
    def __init__(self, ds):
        self.d = ds

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


# ---------------- model ----------------------
class RelPosBias(nn.Module):
    def __init__(self, heads, max_dist=128):
        super().__init__()
        self.rel = nn.Embedding(2 * max_dist, heads)
        self.max_dist = max_dist

    def forward__(self, q, k):
        ctx = torch.arange(k)[None] - torch.arange(q)[:, None]
        ctx = ctx.clamp(-self.max_dist, self.max_dist) + self.max_dist
        return self.rel(ctx)  # [q,k,h]

    def forward(self, q, k):
        return self.forward__(q, k).permute(2, 0, 1)


class CharTransformer(nn.Module):
    def __init__(self, vocab, d=128, heads=8, layers=4, num_cls=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d))
        self.encs = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, num_cls)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos[:, : x.size(1)]
        for layer in self.encs:
            h = layer(h, src_key_padding_mask=mask)
        h = self.norm(h)
        return self.fc(h[:, 0])  # CLS


# ---------------- utils ----------------------
def cwa(pred, lab, w):
    return ((pred == lab).astype(float) * w).sum() / w.sum()


def curriculum(epoch, total):
    return min(1.0, (epoch + 1) / (total / 2))


# --------------- training setup --------------
batch_size, epochs = 32, 12
model = CharTransformer(vsize).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # ← ablation: no smoothing
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

experiment_data = {
    "no_label_smoothing": {
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
for ep in range(epochs):
    model.train()
    running_loss, n_items = 0.0, 0
    cur_w = curriculum(ep, epochs)
    for bd in train_loader:
        bd = {k: v.to(device) for k, v in bd.items()}
        opt.zero_grad()
        logits = model(bd["input_ids"])
        loss = criterion(logits, bd["labels"])
        loss = (loss * torch.where(bd["weights"] > 5, cur_w, 1.0)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running_loss += loss.item() * bd["labels"].size(0)
        n_items += bd["labels"].size(0)
    train_loss = running_loss / n_items
    experiment_data["no_label_smoothing"]["SPR_BENCH"]["losses"]["train"].append(
        train_loss
    )

    # ----- validation -----
    model.eval()
    v_loss, v_count = 0.0, 0
    preds, labs, wts = [], [], []
    with torch.no_grad():
        for bd in dev_loader:
            bd = {k: v.to(device) for k, v in bd.items()}
            out = model(bd["input_ids"])
            loss = criterion(out, bd["labels"])
            v_loss += loss.item() * bd["labels"].size(0)
            v_count += bd["labels"].size(0)
            p = out.argmax(1).cpu().numpy()
            l = bd["labels"].cpu().numpy()
            w = bd["weights"].cpu().numpy()
            preds.extend(p)
            labs.extend(l)
            wts.extend(w)
    v_loss /= v_count
    mf1 = f1_score(labs, preds, average="macro")
    cw = cwa(np.array(preds), np.array(labs), np.array(wts))
    experiment_data["no_label_smoothing"]["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["no_label_smoothing"]["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": mf1, "cwa": cw}
    )
    print(f"Epoch {ep+1}: val_loss={v_loss:.4f} | Macro-F1={mf1:.3f} | CWA={cw:.3f}")

    # early stopping
    if v_loss < best_val - 1e-4:
        best_val, bad_epochs = v_loss, 0
    else:
        bad_epochs += 1
    if bad_epochs >= 3:
        print("Early stopping.")
        break
    sched.step()

experiment_data["no_label_smoothing"]["SPR_BENCH"]["predictions"] = preds
experiment_data["no_label_smoothing"]["SPR_BENCH"]["ground_truth"] = labs
experiment_data["no_label_smoothing"]["SPR_BENCH"]["weights"] = wts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
