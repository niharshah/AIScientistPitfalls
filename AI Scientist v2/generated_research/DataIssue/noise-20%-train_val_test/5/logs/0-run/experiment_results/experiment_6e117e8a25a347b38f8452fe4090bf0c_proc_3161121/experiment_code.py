import os, pathlib, random, numpy as np, torch, math, time, json
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ----------------- book-keeping ----------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "learning_rate": {
        "SPR_BENCH": {
            "lr_values": [],
            "metrics": {
                "train_acc": [],
                "val_acc": [],
                "train_loss": [],
                "val_loss": [],
            },
            "test_acc": [],
            "predictions": [],
            "ground_truth": [],  # filled once
            "epochs": [],
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- utils ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
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


def build_synthetic(n_train=500, n_dev=100, n_test=200, seqlen=10, vocab_sz=12):
    tokens = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(tokens) for _ in range(seqlen)]
            lbl = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(lbl)
        return Dataset.from_dict(data)

    return DatasetDict(train=gen(n_train), dev=gen(n_dev), test=gen(n_test))


def build_vocab(dataset: Dataset, field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[field]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.split()]
    return toks[:max_len] if max_len else toks


def collate(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    maxlen = max(map(len, seqs))
    padded = [s + [vocab["<pad>"]] * (maxlen - len(s)) for s in seqs]
    x = torch.tensor(padded)
    mask = x == vocab["<pad>"]
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# ---------------- model ---------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_sz, emb, nhead, layers, num_cls, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, emb)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=emb * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Linear(emb, num_cls)

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.embed(ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        m = (~mask).unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.cls(pooled)


def evaluate(model, dl, criterion):
    model.eval()
    tot_loss = tot_corr = tot = 0
    with torch.no_grad():
        for bt in dl:
            bt = {k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)}
            out = model(bt["input_ids"], bt["attention_mask"])
            loss = criterion(out, bt["labels"])
            tot_loss += loss.item() * bt["labels"].size(0)
            pred = out.argmax(-1)
            tot_corr += (pred == bt["labels"]).sum().item()
            tot += bt["labels"].size(0)
    return tot_loss / tot, tot_corr / tot


# ---------------- data load ---------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded real dataset.")
except Exception as e:
    print("Falling back to synthetic:", e)
    dsets = build_synthetic()

vocab = build_vocab(dsets["train"])
num_cls = len(set(dsets["train"]["label"]))
batch_size = 64
train_dl = DataLoader(
    dsets["train"], batch_size, True, collate_fn=lambda b: collate(b, vocab)
)
dev_dl = DataLoader(
    dsets["dev"], batch_size, False, collate_fn=lambda b: collate(b, vocab)
)
test_dl = DataLoader(
    dsets["test"], batch_size, False, collate_fn=lambda b: collate(b, vocab)
)

# ground truth once
experiment_data["learning_rate"]["SPR_BENCH"]["ground_truth"] = [
    x["label"] for x in dsets["test"]
]

learning_rates = [5e-4, 1e-3, 2e-3, 3e-3]
epochs = 5
criterion = nn.CrossEntropyLoss()

for lr in learning_rates:
    print(f"\n=== training with lr={lr} ===")
    model = SimpleTransformerClassifier(
        len(vocab), 128, 4, 2, num_cls, vocab["<pad>"]
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # log containers
    tr_acc_list, va_acc_list, tr_loss_list, va_loss_list = [], [], [], []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = tot_corr = tot = 0
        for bt in train_dl:
            bt = {k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)}
            optim.zero_grad()
            out = model(bt["input_ids"], bt["attention_mask"])
            loss = criterion(out, bt["labels"])
            loss.backward()
            optim.step()
            tot_loss += loss.item() * bt["labels"].size(0)
            tot_corr += (out.argmax(-1) == bt["labels"]).sum().item()
            tot += bt["labels"].size(0)
        tr_loss = tot_loss / tot
        tr_acc = tot_corr / tot
        va_loss, va_acc = evaluate(model, dev_dl, criterion)
        print(f"  Epoch {ep}: val_acc={va_acc:.4f} train_acc={tr_acc:.4f}")
        tr_loss_list.append(tr_loss)
        va_loss_list.append(va_loss)
        tr_acc_list.append(tr_acc)
        va_acc_list.append(va_acc)

    # test eval
    test_loss, test_acc = evaluate(model, test_dl, criterion)
    print(f"  Test acc={test_acc:.4f}")

    # predictions
    preds = []
    model.eval()
    with torch.no_grad():
        for bt in test_dl:
            b = {k: v.to(device) for k, v in bt.items() if isinstance(v, torch.Tensor)}
            preds.extend(
                model(b["input_ids"], b["attention_mask"]).argmax(-1).cpu().tolist()
            )

    # store
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
    ed["lr_values"].append(lr)
    ed["metrics"]["train_acc"].append(tr_acc_list)
    ed["metrics"]["val_acc"].append(va_acc_list)
    ed["metrics"]["train_loss"].append(tr_loss_list)
    ed["metrics"]["val_loss"].append(va_loss_list)
    ed["test_acc"].append(test_acc)
    ed["predictions"].append(preds)
    ed["epochs"] = list(range(1, epochs + 1))

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to", os.path.join(working_dir, "experiment_data.npy"))
