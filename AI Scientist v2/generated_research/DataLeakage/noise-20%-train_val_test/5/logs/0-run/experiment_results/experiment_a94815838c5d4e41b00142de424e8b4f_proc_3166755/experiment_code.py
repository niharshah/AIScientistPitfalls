import os, pathlib, time, math, json, torch, numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

# ------------------------ WORKING DIR & DEVICE ---------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------- DATA --------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _one(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    dd["train"] = _one("train.csv")
    dd["dev"] = _one("dev.csv")
    dd["test"] = _one("test.csv")
    return dd


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    idx = 3
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode(seq, vocab, max_len=None):
    toks = [vocab["<cls>"]] + [
        vocab.get(t, vocab["<unk>"]) for t in seq.strip().split()
    ]
    return toks[:max_len] if max_len else toks


def collate(batch, vocab, max_len=128):
    specials = {vocab["<pad>"], vocab["<unk>"], vocab["<cls>"]}
    seqs = [encode(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    L = max(len(s) for s in seqs)
    padded, masks, counts = [], [], []
    for s in seqs:
        pad_len = L - len(s)
        padded_seq = s + [vocab["<pad>"]] * pad_len
        mask = [0] * len(s) + [1] * pad_len
        cnt = np.zeros(len(vocab), dtype=np.float32)
        for tid in s:
            if tid not in specials:
                cnt[tid] += 1.0
        padded.append(padded_seq)
        masks.append(mask)
        counts.append(cnt)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
        "counts": torch.tensor(np.stack(counts), dtype=torch.float32),
        "labels": labels,
    }


# --------------------------- MODEL ---------------------------------- #
class CountAugTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.count_proj = nn.Linear(vocab_size, embed_dim)
        self.cls = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, ids, mask, counts):
        B, L = ids.size()
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(ids) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        pooled = h[:, 0]  # CLS token
        cnt_emb = torch.relu(self.count_proj(counts))
        feats = torch.cat([pooled, cnt_emb], dim=-1)
        return self.cls(feats)


# ----------------------- LOAD DATASET ------------------------------- #
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("SPR_BENCH not found, generating tiny synthetic data.", e)

    def synth(n):
        return ["A B C"] * n

    d = {"id": list(range(500)), "sequence": synth(500), "label": [0] * 500}
    dsets = DatasetDict(
        train=Dataset.from_dict(d), dev=Dataset.from_dict(d), test=Dataset.from_dict(d)
    )

vocab = build_vocab(dsets["train"])
pad_idx = vocab["<pad>"]
num_classes = len(set(dsets["train"]["label"]))
print(f"Vocab size {len(vocab)}, classes {num_classes}")

# ----------------------- DATALOADERS -------------------------------- #
BATCH = 128
train_dl = DataLoader(
    dsets["train"],
    batch_size=BATCH,
    shuffle=True,
    collate_fn=lambda b: collate(b, vocab),
)
dev_dl = DataLoader(
    dsets["dev"],
    batch_size=BATCH,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)
test_dl = DataLoader(
    dsets["test"],
    batch_size=BATCH,
    shuffle=False,
    collate_fn=lambda b: collate(b, vocab),
)

# ---------------------- EXPERIMENT DATA ----------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ----------------------- TRAINING LOOP ------------------------------ #
model = CountAugTransformer(
    vocab_size=len(vocab),
    embed_dim=128,
    nhead=8,
    num_layers=4,
    num_classes=num_classes,
    pad_idx=pad_idx,
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_f1, best_state = 0.0, None
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss = tot_correct = tot_count = 0
    for batch in train_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        opt.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        opt.step()
        with torch.no_grad():
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            tot_correct += (preds == batch["labels"]).sum().item()
            tot_count += batch["labels"].size(0)
    train_loss = tot_loss / tot_count
    train_acc = tot_correct / tot_count

    # ---- validate ----
    model.eval()
    v_loss = v_correct = v_cnt = 0
    all_preds = []
    all_lbls = []
    with torch.no_grad():
        for batch in dev_dl:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
            loss = criterion(logits, batch["labels"])
            v_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            v_correct += (preds == batch["labels"]).sum().item()
            v_cnt += batch["labels"].size(0)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(batch["labels"].cpu().numpy())
    val_loss = v_loss / v_cnt
    val_acc = v_correct / v_cnt
    val_f1 = f1_score(all_lbls, all_preds, average="macro")

    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macroF1 = {val_f1:.4f}"
    )

    # save metrics
    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------------------ TEST EVAL --------------------------------- #
model.load_state_dict(best_state)
model.to(device)
model.eval()
all_preds, all_lbls = [], []
t_loss = t_correct = t_cnt = 0
with torch.no_grad():
    for batch in test_dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"], batch["counts"])
        loss = criterion(logits, batch["labels"])
        t_loss += loss.item() * batch["labels"].size(0)
        preds = logits.argmax(-1)
        t_correct += (preds == batch["labels"]).sum().item()
        t_cnt += batch["labels"].size(0)
        all_preds.extend(preds.cpu().numpy())
        all_lbls.extend(batch["labels"].cpu().numpy())
test_loss = t_loss / t_cnt
test_acc = t_correct / t_cnt
test_f1 = f1_score(all_lbls, all_preds, average="macro")
print(f"Test accuracy = {test_acc:.4f}, Test macro-F1 = {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = all_preds
experiment_data["SPR_BENCH"]["ground_truth"] = all_lbls
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
