import os, pathlib, random, time, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- mandatory working dir & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- reproducibility -----------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# -------------------- dataset loading -----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _l(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- vocab building ------------------------------------
special = ["<PAD>", "<CLS>"]
chars = sorted({c for s in spr["train"]["sequence"] for c in s})
itos = special + chars
stoi = {c: i for i, c in enumerate(itos)}
pad_id, cls_id = stoi["<PAD>"], stoi["<CLS>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab={vocab_size}, classes={num_classes}")


# -------------------- torch dataset -------------------------------------
class SPRTorchDS(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        ids = [cls_id] + [stoi[c] for c in self.seq[idx]]
        counts = torch.bincount(torch.tensor(ids[1:]), minlength=vocab_size).float()
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "counts": counts,
            "label": torch.tensor(self.lab[idx], dtype=torch.long),
        }


def collate(batch):
    inps = [b["input_ids"] for b in batch]
    counts = torch.stack([b["counts"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    inps = pad_sequence(inps, batch_first=True, padding_value=pad_id)
    return {"input_ids": inps, "counts": counts, "label": labels}


bs = 128
train_loader = DataLoader(
    SPRTorchDS(spr["train"]), bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(SPRTorchDS(spr["dev"]), bs, shuffle=False, collate_fn=collate)
test_loader = DataLoader(SPRTorchDS(spr["test"]), bs, shuffle=False, collate_fn=collate)


# -------------------- model ---------------------------------------------
class HybridCountTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, n_cls, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(1024, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        # count branch
        self.count_proj = nn.Sequential(
            nn.Linear(vocab, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        # classifier
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(d_model, n_cls)

    def forward(self, x, counts):
        mask = x.eq(pad_id)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_h = h[:, 0]  # CLS token
        cnt_h = self.count_proj(counts)
        fused = self.dropout(cls_h + cnt_h)  # simple fusion
        return self.cls(fused)


model = HybridCountTransformer(vocab_size, 256, 8, 4, num_classes, pad_id).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
steps_ep = math.ceil(len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, steps_per_epoch=steps_ep, epochs=15, pct_start=0.3
)

# -------------------- experiment data dict ------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- helpers -------------------------------------------
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    scaler = torch.cuda.amp.GradScaler(enabled=training)
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast():
            logits = model(batch["input_ids"], batch["counts"])
            loss = criterion(logits, batch["label"])
        if training:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# -------------------- training loop -------------------------------------
best_f1, patience, wait = -1.0, 5, 0
for epoch in range(1, 16):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_f1 = {val_f1:.4f}")
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(tr_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)
    if val_f1 > best_f1:
        best_f1, wait = val_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
    print(f"Epoch time: {time.time()-t0:.1f}s")

# -------------------- test evaluation -----------------------------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"TEST: loss={test_loss:.4f}, macro_f1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# -------------------- save ----------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
