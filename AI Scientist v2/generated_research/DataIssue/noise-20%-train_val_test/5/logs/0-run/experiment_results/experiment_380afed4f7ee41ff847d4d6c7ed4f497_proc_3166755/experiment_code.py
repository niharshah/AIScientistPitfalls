import os, pathlib, random, numpy as np, torch, math
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------ SET-UP & GLOBALS ------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------ DATA  HELPERS --------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_synthetic(n_train=5000, n_dev=1000, n_test=2000, seqlen=12, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def make(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(int(seq.count("A") % 2 == 0))
        return Dataset.from_dict(d)

    return DatasetDict(train=make(n_train), dev=make(n_dev), test=make(n_test))


# try load real dataset
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic:", e)
    dsets = build_synthetic()


# vocab with CLS
def build_vocab(dataset, seq_field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
    idx = 3
    for s in dataset[seq_field]:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(dsets["train"])
PAD, UNK, CLS = vocab["<pad>"], vocab["<unk>"], vocab["<cls>"]
print("Vocab size:", len(vocab))


def encode(seq):
    ids = [CLS] + [vocab.get(t, UNK) for t in seq.split()]
    return ids


def collate(batch, max_len=128):
    xs = [encode(b["sequence"])[:max_len] for b in batch]
    maxL = max(len(x) for x in xs)
    xs = [x + [PAD] * (maxL - len(x)) for x in xs]
    x = torch.tensor(xs, dtype=torch.long)
    mask = x.eq(PAD)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"input_ids": x, "attention_mask": mask, "labels": y}


batch_size = 64
train_dl = DataLoader(dsets["train"], batch_size, shuffle=True, collate_fn=collate)
dev_dl = DataLoader(dsets["dev"], batch_size, shuffle=False, collate_fn=collate)
test_dl = DataLoader(dsets["test"], batch_size, shuffle=False, collate_fn=collate)


# ------------------ MODEL ----------------------------- #
class CLSTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        nhead=2,
        n_layers=2,
        num_cls=2,
        pad_idx=0,
        max_len=256,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, num_cls)

    def forward(self, x, mask):
        B, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        cls_vec = h[:, 0, :]  # representation of CLS token
        return self.fc(cls_vec)


model = CLSTransformer(len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ------------------ TRAIN / EVAL LOOPS ---------------- #
def run_eval(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            count += batch["labels"].size(0)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / count
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_preds, all_labels


epochs = 6
for epoch in range(1, epochs + 1):
    model.train()
    tot_loss, tot_cnt = 0, 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        tot_cnt += batch["labels"].size(0)
    train_loss = tot_loss / tot_cnt
    _, train_f1, _, _ = run_eval(train_dl)
    val_loss, val_f1, _, _ = run_eval(dev_dl)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_macroF1 = {val_f1:.4f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)

# ------------------ TEST EVALUATION ------------------- #
test_loss, test_f1, preds, gt = run_eval(test_dl)
print(f"Test macro-F1 = {test_f1:.4f}")
experiment_data["SPR_BENCH"]["metrics"]["test_f1"] = test_f1
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gt

# ------------------ SAVE DATA ------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
