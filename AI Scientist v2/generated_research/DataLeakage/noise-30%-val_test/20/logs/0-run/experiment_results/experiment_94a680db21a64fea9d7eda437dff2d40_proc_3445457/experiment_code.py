import os, pathlib, random, string, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    # tiny synthetic fallback
    def synth(n, sid=0):
        rows = []
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:12], k=random.randint(5, 15))
            )
            label = int(seq.count("A") % 2 == 0)
            rows.append({"id": sid + i, "sequence": seq, "label": label})
        return rows

    spr = DatasetDict()
    spr["train"] = load_dataset(
        "json", data_files={"train": synth(3000)}, split="train"
    )
    spr["dev"] = load_dataset(
        "json", data_files={"train": synth(600, 3000)}, split="train"
    )
    spr["test"] = load_dataset(
        "json", data_files={"train": synth(600, 3600)}, split="train"
    )
print({k: len(v) for k, v in spr.items()})

# -------------------------------------------------
vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        vocab.setdefault(ch, len(vocab))
vsz = len(vocab)
max_len = min(max(len(ex["sequence"]) for ex in spr["train"]) + 1, 128)


def encode(seq):
    ids = [vocab["<cls>"]] + [vocab.get(c, 1) for c in seq][: max_len - 1]
    ids += [0] * (max_len - len(ids))
    return ids


def aux_labels(seq):
    # aux1: parity of length (0=even,1=odd)
    aux1 = len(seq) % 2
    # aux2: bucketized unique symbol count: 0:(1-4),1:(5-8),2:(9+)
    u = len(set(seq))
    aux2 = 0 if u <= 4 else 1 if u <= 8 else 2
    return aux1, aux2


class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.d = hf_ds

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        ex = self.d[idx]
        ids = torch.tensor(encode(ex["sequence"]), dtype=torch.long)
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        a1, a2 = aux_labels(ex["sequence"])
        return {
            "input_ids": ids,
            "labels": y,
            "aux1": torch.tensor(a1),
            "aux2": torch.tensor(a2),
            "weights": torch.tensor(float(ex.get("complexity", 1.0))),
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out


train_ds, dev_ds, test_ds = map(
    SPRTorchDataset, (spr["train"], spr["dev"], spr["test"])
)


# -------------------------------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 64
        self.emb = nn.Embedding(vsz, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.main_head = nn.Linear(emb_dim, 2)
        self.aux1_head = nn.Linear(emb_dim, 2)
        self.aux2_head = nn.Linear(emb_dim, 3)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos[: x.size(1)]
        h = self.encoder(h, src_key_padding_mask=mask)
        cls = h[:, 0]
        return self.main_head(cls), self.aux1_head(cls), self.aux2_head(cls)


def cw_accuracy(pred, lab, wt):
    correct = (pred == lab).astype(float)
    return (correct * wt).sum() / wt.sum()


# -------------------------------------------------
batch = 32
epochs = 10
early_patience = 3
model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
ce = nn.CrossEntropyLoss()
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "weights": [],
    }
}

best_cwa = 0
no_improve = 0
for ep in range(1, epochs + 1):
    model.train()
    tl = 0
    n = 0
    for bt in train_loader:
        bt = {k: v.to(device) for k, v in bt.items()}
        opt.zero_grad()
        out_main, out_a1, out_a2 = model(bt["input_ids"])
        loss = (
            ce(out_main, bt["labels"])
            + 0.3 * ce(out_a1, bt["aux1"])
            + 0.3 * ce(out_a2, bt["aux2"])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * bt["labels"].size(0)
        n += bt["labels"].size(0)
    train_loss = tl / n
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

    # validation
    model.eval()
    vl = 0
    vn = 0
    preds = []
    labs = []
    wts = []
    with torch.no_grad():
        for bt in dev_loader:
            bt = {k: v.to(device) for k, v in bt.items()}
            o_main, o_a1, o_a2 = model(bt["input_ids"])
            loss = (
                ce(o_main, bt["labels"])
                + 0.3 * ce(o_a1, bt["aux1"])
                + 0.3 * ce(o_a2, bt["aux2"])
            )
            vl += loss.item() * bt["labels"].size(0)
            vn += bt["labels"].size(0)
            p = o_main.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            labs.extend(bt["labels"].cpu().numpy().tolist())
            wts.extend(bt["weights"].cpu().numpy().tolist())
    val_loss = vl / vn
    macro = f1_score(labs, preds, average="macro")
    cwa = cw_accuracy(np.array(preds), np.array(labs), np.array(wts))
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"macro_f1": macro, "cwa": cwa}
    )
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | Macro-F1={macro:.4f} | CWA={cwa:.4f}"
    )
    if cwa > best_cwa + 1e-4:
        best_cwa = cwa
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= early_patience:
        print("Early stopping.")
        break
    sched.step()

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labs
experiment_data["SPR_BENCH"]["weights"] = wts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
