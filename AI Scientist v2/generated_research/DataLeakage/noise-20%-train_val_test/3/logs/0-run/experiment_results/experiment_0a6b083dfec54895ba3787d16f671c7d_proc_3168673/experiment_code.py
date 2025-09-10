import os, pathlib, random, math, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ----------------------------- setup & reproducibility ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------- data --------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


# adjust path if provided through env else use default
DATA_PATH = pathlib.Path(
    os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# build char-level vocab
special_tokens = ["<PAD>", "<CLS>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, cls_id = stoi["<PAD>"], stoi["<CLS>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"Vocab={vocab_size}, classes={num_classes}")


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [cls_id] + [stoi[c] for c in self.seqs[idx]]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate_fn
)


# ----------------------------- model --------------------------------------------------
class CLSTransformer_NoDropout(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, n_cls, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(512, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.dropout = nn.Identity()  # disabled
        self.cls = nn.Linear(d_model, n_cls)

    def forward(self, x):
        mask = x.eq(pad_id)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_h = self.dropout(h[:, 0])
        return self.cls(cls_h)


model = CLSTransformer_NoDropout(vocab_size, 512, 8, 6, num_classes, pad_id).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
steps_per_epoch = math.ceil(len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, steps_per_epoch=steps_per_epoch, epochs=20, pct_start=0.3
)


# ----------------------------- helpers -----------------------------------------------
def run_epoch(loader, train=True):
    (model.train if train else model.eval)()
    total_loss, preds, gts = 0.0, [], []
    scaler = torch.cuda.amp.GradScaler(enabled=train)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast():
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
        if train:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ----------------------------- train loop --------------------------------------------
experiment_data = {
    "no_dropout": {
        "SPR_BENCH": {
            "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
best_f1, patience, wait = -1.0, 6, 0
for epoch in range(1, 21):
    t0 = time.time()
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, train=False)
    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}"
    )
    ed = experiment_data["no_dropout"]["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(tr_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)
    # early stopping
    if val_f1 > best_f1:
        best_f1, wait = val_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break
    print(f"Epoch time: {time.time()-t0:.1f}s")

# ----------------------------- evaluate best -----------------------------------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, train=False)
print(f"TEST: loss={test_loss:.4f}, macro_f1={test_f1:.4f}")
ed = experiment_data["no_dropout"]["SPR_BENCH"]
ed["predictions"] = test_preds
ed["ground_truth"] = test_gts

# ----------------------------- save ---------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
