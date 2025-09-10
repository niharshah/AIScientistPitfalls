# ------------------------------ Remove-PE Ablation -----------------------------
import os, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# reproducibility --------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# --------------------- working dir & device -----------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- experiment data dict -----------------------------------
experiment_data = {
    "no_positional_encoding": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# --------------------- locate & load dataset ----------------------------------
def _find_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]
    needed = {"train.csv", "dev.csv", "test.csv"}
    for c in candidates:
        if c and c.exists() and needed.issubset({p.name for p in c.iterdir()}):
            print(f"Found SPR_BENCH at {c}")
            return c
    raise FileNotFoundError(
        "SPR_BENCH not found; set SPR_DATA env var or place folder."
    )


root = _find_spr_bench()


def load_spr_bench(path: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(path / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


spr = load_spr_bench(root)


# --------------------- vocabulary & dataset -----------------------------------
class SPRTokenDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.hf = hf_ds
        self.vocab = vocab
        self.pad = vocab["<pad>"]
        self.cls = vocab["<cls>"]

    def __len__(self):
        return len(self.hf)

    def encode(self, s):
        toks = s.strip().split()
        return [self.cls] + [self.vocab[t] for t in toks]

    def __getitem__(self, idx):
        row = self.hf[idx]
        ids = torch.tensor(self.encode(row["sequence"]), dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        parity = torch.tensor((len(ids) - 1) % 2, dtype=torch.long)
        return {"input_ids": ids, "labels": label, "parity": parity}


def build_vocab(train_split):
    vocab = {"<pad>": 0, "<cls>": 1}
    for ex in train_split:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


def collate(batch, pad_id):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    parity = torch.stack([b["parity"] for b in batch])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attn = (padded != pad_id).long()
    return {
        "input_ids": padded,
        "attention_mask": attn,
        "labels": labels,
        "parity": parity,
    }


train_ds = SPRTokenDataset(spr["train"], vocab)
dev_ds = SPRTokenDataset(spr["dev"], vocab)
test_ds = SPRTokenDataset(spr["test"], vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab["<pad>"]),
)

num_labels = len({int(ex["label"]) for ex in spr["train"]})
max_len = max(len(ex["sequence"].split()) + 1 for ex in spr["train"])
print("Max token length:", max_len)


# --------------------- model (WITHOUT positional encoding) --------------------
class SPRTransformerNoPE(nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=256, nhead=8, nlayers=6, ff=512, drop=0.15
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, drop, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.main_head = nn.Linear(d_model, num_labels)
        self.parity_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)  # (B, L, d_model)
        x = self.enc(x, src_key_padding_mask=(attention_mask == 0))
        cls = self.norm(x[:, 0])
        return self.main_head(cls), self.parity_head(cls)


model = SPRTransformerNoPE(len(vocab), num_labels).to(device)

# --------------------- optimiser & loss ---------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
criterion_main = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_parity = nn.CrossEntropyLoss()
aux_weight = 0.2


# --------------------- metric --------------------------------------------------
def macro_f1(preds, labels, num_cls):
    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    f1s = []
    for c in range(num_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# --------------------- train / val epoch --------------------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_cnt = 0
    all_preds, all_lbls = [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        main_logits, parity_logits = model(batch["input_ids"], batch["attention_mask"])
        loss_main = criterion_main(main_logits, batch["labels"])
        loss_parity = criterion_parity(parity_logits, batch["parity"])
        loss = loss_main + aux_weight * loss_parity
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            preds = main_logits.argmax(1)
        tot_loss += loss.item() * len(preds)
        tot_correct += (preds == batch["labels"]).sum().item()
        tot_cnt += len(preds)
        all_preds.append(preds)
        all_lbls.append(batch["labels"])
    all_preds = torch.cat(all_preds)
    all_lbls = torch.cat(all_lbls)
    acc = tot_correct / tot_cnt
    f1 = macro_f1(all_preds, all_lbls, num_labels)
    return tot_loss / tot_cnt, acc, f1


# --------------------- training loop ------------------------------------------
best_f1 = 0
patience = 2
wait = 0
epochs = 10
for epoch in range(1, epochs + 1):
    t_loss, t_acc, t_f1 = run_epoch(train_loader, True)
    v_loss, v_acc, v_f1 = run_epoch(dev_loader, False)
    ed = experiment_data["no_positional_encoding"]["SPR_BENCH"]
    ed["losses"]["train"].append(t_loss)
    ed["losses"]["val"].append(v_loss)
    ed["metrics"]["train_acc"].append(t_acc)
    ed["metrics"]["val_acc"].append(v_acc)
    ed["metrics"]["train_f1"].append(t_f1)
    ed["metrics"]["val_f1"].append(v_f1)
    print(
        f"Epoch {epoch}: val_loss={v_loss:.4f} | val_acc={v_acc*100:.2f}% | val_macroF1={v_f1:.4f}"
    )
    if v_f1 > best_f1:
        best_f1 = v_f1
        wait = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# --------------------- test evaluation ----------------------------------------
model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        logits, _ = model(batch["input_ids"], batch["attention_mask"])
        preds_all.append(logits.argmax(1).cpu())
        labels_all.append(batch["labels"].cpu())
preds_all = torch.cat(preds_all)
labels_all = torch.cat(labels_all)
test_acc = (preds_all == labels_all).float().mean().item()
test_f1 = macro_f1(preds_all, labels_all, num_labels)
print(f"Test accuracy: {test_acc*100:.2f}% | Test macroF1: {test_f1:.4f}")

# store predictions & gt -------------------------------------------------------
ed["predictions"] = preds_all.numpy()
ed["ground_truth"] = labels_all.numpy()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
