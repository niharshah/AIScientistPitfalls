import os, pathlib, time, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- reproducibility ----------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- experiment container ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "val_loss": [],
            "URA": [],
            "SWA": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ---------- helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split()})


def count_color_variety(seq: str) -> int:
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def unseen_rule_accuracy(train_labels_set, y_true, y_pred):
    idx = [i for i, l in enumerate(y_true) if l not in train_labels_set]
    if not idx:
        return float("nan")
    return np.mean([y_true[i] == y_pred[i] for i in idx])


# ---------- load data ----------
SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
vocab = {"<pad>": 0, "<unk>": 1}
for seq in spr["train"]["sequence"]:
    for tok in seq.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
v2i = vocab
i2v = [None] * len(vocab)
for k, v in v2i.items():
    i2v[v] = k


def encode(seq: str):
    return [v2i.get(tok, 1) for tok in seq.split()]


train_labels = sorted(set(spr["train"]["label"]))
label2id = {
    l: i for i, l in enumerate(train_labels)
}  # unknown label will be handled later
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id) + 1  # +1 for unseen label token

UNK_LABEL_ID = num_labels - 1


# ---------- dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split, train_mode=True):
        self.seqs = [encode(s) for s in split["sequence"]]
        self.shape_var = [count_shape_variety(s) for s in split["sequence"]]
        self.color_var = [count_color_variety(s) for s in split["sequence"]]
        self.labels = split["label"]
        self.train_mode = train_mode

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        shp = torch.tensor(self.shape_var[idx], dtype=torch.float)
        col = torch.tensor(self.color_var[idx], dtype=torch.float)
        if self.train_mode:
            y = torch.tensor(label2id[self.labels[idx]], dtype=torch.long)
            return {"input": x, "shape": shp, "color": col, "label": y}
        else:
            return {
                "input": x,
                "shape": shp,
                "color": col,
                "label_str": self.labels[idx],
            }


def collate(batch):
    xs = [b["input"] for b in batch]
    lens = [len(x) for x in xs]
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    shapes = torch.tensor([b["shape"] for b in batch], dtype=torch.float)
    colors = torch.tensor([b["color"] for b in batch], dtype=torch.float)
    out = {
        "input": xs_pad,
        "lengths": torch.tensor(lens, dtype=torch.long),
        "shape": shapes,
        "color": colors,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    else:
        out["label_str"] = [b["label_str"] for b in batch]
    return out


train_ds = SPRTorchDataset(spr["train"], True)
dev_ds = SPRTorchDataset(spr["dev"], True)
test_ds = SPRTorchDataset(spr["test"], False)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ---------- model ----------
class HybridSPR(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.mlp_sym = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16))
        self.classifier = nn.Linear(emb_dim + 16, num_labels)

    def forward(self, x, lens, shape_var, color_var):
        e = self.emb(x)  # (B,L,E)
        mask = x == 0  # pad mask
        h = self.encoder(e, src_key_padding_mask=mask)  # (B,L,E)
        # masked mean pooling
        mask_inv = (~mask).unsqueeze(-1)
        h_sum = (h * mask_inv).sum(1)
        h_mean = h_sum / torch.clamp(mask_inv.sum(1), min=1)
        sym_feat = self.mlp_sym(torch.stack([shape_var, color_var], dim=1))
        logits = self.classifier(torch.cat([h_mean, sym_feat], dim=-1))
        return logits


# ---------- training helpers ----------
def run_epoch(model, loader, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = tot_ok = tot = 0
    for batch in loader:
        inp = batch["input"].to(device)
        lens = batch["lengths"].to(device)
        shp = batch["shape"].to(device)
        col = batch["color"].to(device)
        lbl = batch["label"].to(device)
        logits = model(inp, lens, shp, col)
        loss = criterion(logits, lbl)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * inp.size(0)
        tot_ok += (logits.argmax(1) == lbl).sum().item()
        tot += inp.size(0)
    return tot_loss / tot, tot_ok / tot


def evaluate_test(model):
    model.eval()
    preds_all = []
    labels_all = []
    seqs = []
    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input"].to(device)
            lens = batch["lengths"].to(device)
            shp = batch["shape"].to(device)
            col = batch["color"].to(device)
            logits = model(inp, lens, shp, col)
            preds = logits.argmax(1).cpu().tolist()
            preds_lbl = [id2label.get(p, "UNK") for p in preds]
            preds_all.extend(preds_lbl)
            labels_all.extend(batch["label_str"])
            for seq in batch["input"]:
                tokens = [i2v[tok] for tok in seq.tolist() if tok != 0]
                seqs.append(" ".join(tokens))
    swa = shape_weighted_accuracy(seqs, labels_all, preds_all)
    ura = unseen_rule_accuracy(set(train_labels), labels_all, preds_all)
    acc = np.mean([p == t for p, t in zip(preds_all, labels_all)])
    return acc, swa, ura, preds_all, labels_all


# ---------- training ----------
model = HybridSPR(len(vocab), emb_dim=128, hidden_dim=256, num_labels=num_labels).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, dev_loader, criterion)
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc = {val_acc:.4f}")

# ---------- final evaluation ----------
acc, swa, ura, preds, labels = evaluate_test(model)
print(f"\nTEST  Acc={acc:.4f} | SWA={swa:.4f} | URA={ura:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
experiment_data["SPR_BENCH"]["metrics"]["URA"].append(ura)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = labels

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
