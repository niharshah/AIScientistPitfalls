import os, pathlib, random, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------- WORKDIR & DEVICE ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- EXPERIMENT DATA ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- DATA UTILITIES ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def build_vocab(dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    idx = 3
    for s in dataset[seq_field]:
        toks = s.strip().split()
        if len(toks) == 1:  # fallback: treat each char
            toks = list(s.strip())
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = seq.strip().split()
    if len(toks) == 1:  # char-level fallback
        toks = list(seq.strip())
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
    if max_len:
        ids = ids[:max_len]
    return ids


def collate_fn(batch, vocab, max_len=128):
    cls_id = vocab["<cls>"]
    enc = [[cls_id] + encode_sequence(b["sequence"], vocab, max_len - 1) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    maxL = max(len(e) for e in enc)
    pad_id = vocab["<pad>"]
    padded = [e + [pad_id] * (maxL - len(e)) for e in enc]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x.eq(pad_id)
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# ---------- MODEL ----------
class CLS_Transformer(nn.Module):
    def __init__(self, vocab_sz, emb_dim, nhead, nlayers, nclass, pad_idx, max_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )  # pre-LN
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls_head = nn.Linear(emb_dim, nclass)

    def forward(self, ids, mask):
        B, L = ids.shape
        pos = torch.arange(0, L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(ids) + self.pos_emb(pos)
        h = self.encoder(x, src_key_padding_mask=mask)
        cls_vec = h[:, 0]  # first position ([CLS])
        return self.cls_head(cls_vec)


# ---------- DATA LOADING ----------
SPR_ROOT = pathlib.Path(
    os.environ.get("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
try:
    dsets = load_spr_bench(SPR_ROOT)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Falling back to synthetic data.", e)

    def build_syn(n):
        data = {"id": [], "sequence": [], "label": []}
        vocab_sym = [chr(ord("A") + i) for i in range(12)]
        for i in range(n):
            seq = [random.choice(vocab_sym) for _ in range(10)]
            lbl = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(lbl)
        return Dataset.from_dict(data)

    dsets = DatasetDict(train=build_syn(1000), dev=build_syn(200), test=build_syn(500))

vocab = build_vocab(dsets["train"])
print("Vocab size:", len(vocab))

batch_size = 64
train_dl = DataLoader(
    dsets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)

# ---------- TRAINING ----------
model = CLS_Transformer(
    vocab_sz=len(vocab),
    emb_dim=128,
    nhead=2,
    nlayers=3,
    nclass=len(set(dsets["train"]["label"])),
    pad_idx=vocab["<pad>"],
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def run_eval(dloader):
    model.eval()
    losses, preds_all, gts_all = [], [], []
    with torch.no_grad():
        for batch in dloader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            losses.append(loss.item() * batch["labels"].size(0))
            preds = logits.argmax(-1)
            preds_all.extend(preds.cpu().tolist())
            gts_all.extend(batch["labels"].cpu().tolist())
    avg_loss = sum(losses) / len(gts_all)
    acc = np.mean(np.array(preds_all) == np.array(gts_all))
    f1 = f1_score(gts_all, preds_all, average="macro")
    return avg_loss, acc, f1, preds_all, gts_all


epochs = 10
for ep in range(1, epochs + 1):
    model.train()
    tr_losses, tr_preds, tr_gts = [], [], []
    for batch in train_dl:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        opt.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        opt.step()
        tr_losses.append(loss.item() * batch["labels"].size(0))
        preds = logits.argmax(-1)
        tr_preds.extend(preds.detach().cpu().tolist())
        tr_gts.extend(batch["labels"].cpu().tolist())

    train_loss = sum(tr_losses) / len(tr_gts)
    train_acc = np.mean(np.array(tr_preds) == np.array(tr_gts))
    train_f1 = f1_score(tr_gts, tr_preds, average="macro")

    val_loss, val_acc, val_f1, _, _ = run_eval(dev_dl)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
    )

    experiment_data["SPR_BENCH"]["metrics"]["train"].append(
        {"acc": train_acc, "f1": train_f1}
    )
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": val_acc, "f1": val_f1}
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

# ---------- TEST EVAL ----------
test_loss, test_acc, test_f1, preds_all, gts_all = run_eval(test_dl)
print(f"\nTEST  |  loss={test_loss:.4f}  acc={test_acc:.4f}  macroF1={test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds_all
experiment_data["SPR_BENCH"]["ground_truth"] = gts_all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
