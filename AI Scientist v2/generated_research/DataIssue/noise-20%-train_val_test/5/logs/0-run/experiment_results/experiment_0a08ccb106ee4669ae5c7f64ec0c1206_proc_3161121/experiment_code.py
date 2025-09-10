# ---------------------------------------------------------------
# Hyper-parameter tuning: embedding dimension
# single-file, self-contained script
# ---------------------------------------------------------------
import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ---------- mandatory experiment dict (global) -----------------
experiment_data = {
    "embed_dim_tuning": {"SPR_BENCH": {"runs": []}}  # list of dicts (one per embed_dim)
}


# ---------- utility ------------------------------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA UTILS ------------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_vocab(dataset: Dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    return toks[:max_len] if max_len else toks


# ------------------------ synthetic fallback -------------------------
def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen_split(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return Dataset.from_dict(data)

    return DatasetDict(
        train=gen_split(num_train), dev=gen_split(num_dev), test=gen_split(num_test)
    )


# --------------------------- MODEL ---------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, pad_mask):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        mask_flt = (~pad_mask).unsqueeze(-1)  # False = token
        summed = (h * mask_flt).sum(1)
        lens = mask_flt.sum(1).clamp(min=1)
        pooled = summed / lens
        return self.classifier(pooled)


# ----------------------- collate / evaluate --------------------------
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
    return total_loss / count, correct / count


# ------------------- load dataset (real or synthetic) ----------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Real dataset not found, using synthetic.", e)
    datasets_dict = build_synthetic()

vocab = build_vocab(datasets_dict["train"])
num_classes = len(set(datasets_dict["train"]["label"]))
print(f"Vocab size = {len(vocab)}, num_classes = {num_classes}")

batch_size = 64
train_dl = DataLoader(
    datasets_dict["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab),
)
dev_dl = DataLoader(
    datasets_dict["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)
test_dl = DataLoader(
    datasets_dict["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab),
)

# ---------------- hyper-parameter grid (embed_dim) -------------------
embed_dims = [64, 128, 256]  # try small → large
epochs = 5

for ed in embed_dims:
    print(f"\n=== Training embed_dim={ed} ===")
    # pick a reasonable nhead that divides embed_dim
    nhead = max(2, ed // 32)
    while ed % nhead != 0:
        nhead -= 1
    model = SimpleTransformerClassifier(
        len(vocab),
        ed,
        nhead,
        num_layers=2,
        num_classes=num_classes,
        pad_idx=vocab["<pad>"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_record = {
        "embed_dim": ed,
        "epoch": [],
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "test_acc": None,
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        sum_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        train_loss = sum_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(f"embed={ed} Epoch {ep}: val_acc={val_acc:.4f} train_acc={train_acc:.4f}")
        # log
        run_record["epoch"].append(ep)
        run_record["metrics"]["train_acc"].append(train_acc)
        run_record["metrics"]["val_acc"].append(val_acc)
        run_record["losses"]["train"].append(train_loss)
        run_record["losses"]["val"].append(val_loss)

    # --------- final test evaluation for this embed_dim --------------
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"embed={ed} TEST accuracy = {test_acc:.4f}")
    run_record["test_acc"] = test_acc

    # store predictions / gts
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            gts_all.extend(batch["labels"])
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds_all.extend(logits.argmax(-1).cpu().tolist())
    run_record["predictions"] = preds_all
    run_record["ground_truth"] = gts_all

    experiment_data["embed_dim_tuning"]["SPR_BENCH"]["runs"].append(run_record)

# ---------------- save everything ------------------------------------
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
