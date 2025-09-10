import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ---------------------------  I/O & SEED  ---------------------------- #
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"optimizer_weight_decay": {"SPR_BENCH": {}}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA UTILS ------------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # helper
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_vocab(dataset: Dataset, seq_field: str = "sequence"):
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


def build_synthetic(nt=500, nd=100, nte=200, seqlen=10, vsz=12):
    syms = [chr(ord("A") + i) for i in range(vsz)]

    def gensplit(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            lab = 1 if seq.count("A") % 2 == 0 else 0
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(lab)
        return Dataset.from_dict(d)

    return DatasetDict(train=gensplit(nt), dev=gensplit(nd), test=gensplit(nte))


# --------------------------- MODEL ----------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, layers, n_cls, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim,
            nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.classifier = nn.Linear(embed_dim, n_cls)

    def forward(self, x, mask):
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1).float()
        h_sum = (h * mask_inv).sum(1)
        pooled = h_sum / mask_inv.sum(1).clamp(min=1)
        return self.classifier(pooled)


# ------------------------ COLLATE & EVAL ----------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    ml = max(len(s) for s in seqs)
    padded = [s + [vocab["<pad>"]] * (ml - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, dl, crit):
    model.eval()
    tot_loss = tot_corr = tot = 0
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = crit(out, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = out.argmax(-1)
            tot_corr += (preds == batch["labels"]).sum().item()
            tot += batch["labels"].size(0)
    return tot_loss / tot, tot_corr / tot


# ------------------------- LOAD DATA --------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, using synthetic instead:", e)
    datasets_dict = build_synthetic()

vocab = build_vocab(datasets_dict["train"])
num_cls = len(set(datasets_dict["train"]["label"]))
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

# ---------------------  HYPERPARAMETER SWEEP  ------------------------ #
weight_decays = [0.0, 1e-4, 1e-3, 1e-2]
epochs = 5
best_val = -1
best_run_pred, best_run_gt = None, None

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = SimpleTransformerClassifier(
        len(vocab), 128, 4, 2, num_cls, vocab["<pad>"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_metrics = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = tot_corr = tot = 0
        for batch in train_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            tot_corr += (logits.argmax(-1) == batch["labels"]).sum().item()
            tot += batch["labels"].size(0)
        train_loss = tot_loss / tot
        train_acc = tot_corr / tot
        val_loss, val_acc = evaluate(model, dev_dl, criterion)
        print(f"Ep {ep}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        run_metrics["train_acc"].append(train_acc)
        run_metrics["val_acc"].append(val_acc)
        run_metrics["train_loss"].append(train_loss)
        run_metrics["val_loss"].append(val_loss)

    # test evaluation & predictions
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            cpu_labels = batch["labels"]
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["input_ids"], batch["attention_mask"])
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(cpu_labels.tolist())

    print(f"Weight_decay {wd}: test_acc={test_acc:.4f}")
    # store
    key = f"wd_{wd}"
    experiment_data["optimizer_weight_decay"]["SPR_BENCH"][key] = {
        "metrics": run_metrics,
        "test_acc": test_acc,
        "predictions": preds,
        "ground_truth": gts,
    }

    # track best run
    if run_metrics["val_acc"][-1] > best_val:
        best_val = run_metrics["val_acc"][-1]
        best_run_pred, best_run_gt = preds, gts

# additional convenience store of best run
experiment_data["optimizer_weight_decay"]["SPR_BENCH"]["best_run"] = {
    "val_acc": best_val,
    "predictions": best_run_pred,
    "ground_truth": best_run_gt,
}

# ---------------------- SAVE RESULTS --------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
