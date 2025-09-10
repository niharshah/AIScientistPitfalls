import os, random, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ------------------------------------------------------------------ #
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------ #
# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------ #
# experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------ #
# dataset utilities
def make_synthetic_split(size: int = 1000):
    toks = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    seqs, labels = [], []
    for i in range(size):
        length = random.randint(3, 12)
        seq = " ".join(random.choice(toks) for _ in range(length))
        label = "even" if length % 2 == 0 else "odd"
        seqs.append(seq)
        labels.append(label)
    return HFDataset.from_dict(
        {"id": list(range(size)), "sequence": seqs, "label": labels}
    )


def load_spr_bench() -> DatasetDict:
    root = pathlib.Path(os.getenv("SPR_DATA_PATH", "SPR_BENCH"))
    train_csv, dev_csv, test_csv = (
        root / "train.csv",
        root / "dev.csv",
        root / "test.csv",
    )

    if train_csv.exists() and dev_csv.exists() and test_csv.exists():

        def _load(csv_path):
            return load_dataset(
                "csv",
                data_files=str(csv_path),
                split="train",
                cache_dir=".cache_dsets",
            )

        print(f"Loading dataset from {root}")
        return DatasetDict(
            train=_load(train_csv), dev=_load(dev_csv), test=_load(test_csv)
        )

    print("SPR_BENCH csv files not found – creating synthetic dataset")
    return DatasetDict(
        train=make_synthetic_split(1000),
        dev=make_synthetic_split(200),
        test=make_synthetic_split(200),
    )


dsets = load_spr_bench()
print({k: len(v) for k, v in dsets.items()})

# ------------------------------------------------------------------ #
# vocabulary & encoding helpers
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}
for split in dsets.values():
    for seq in split["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)

labels = sorted(set(dsets["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
MAX_LEN = 64


def encode_sequence(seq: str):
    toks = seq.strip().split()
    ids = [vocab[CLS]] + [vocab.get(t, vocab[UNK]) for t in toks]
    ids = ids[:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


BATCH = 64
train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)


# ------------------------------------------------------------------ #
# model definitions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BaseTransformer(nn.Module):
    def __init__(self, use_positional=True, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.use_positional = use_positional
        if use_positional:
            self.positional = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layer)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn_mask):
        x = self.emb(ids)
        if self.use_positional:
            x = self.positional(x)
        x = self.enc(x, src_key_padding_mask=~attn_mask.bool())
        return self.cls(x[:, 0])


criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, preds_all, gts_all = 0.0, [], []

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        preds_all.extend(preds)
        gts_all.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts_all, preds_all, average="macro")
    acc = accuracy_score(gts_all, preds_all)
    mcc = matthews_corrcoef(gts_all, preds_all)
    return avg_loss, macro_f1, acc, mcc, preds_all, gts_all


def train(model, epochs=2, lr=3e-4, tag="baseline"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, tr_acc, tr_mcc, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_f1, val_acc, val_mcc, preds, gts = run_epoch(model, dev_loader)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(
            {"epoch": epoch, "loss": tr_loss}
        )
        experiment_data["SPR_BENCH"]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_loss}
        )
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            {"epoch": epoch, "macro_f1": tr_f1, "acc": tr_acc, "mcc": tr_mcc}
        )
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "macro_f1": val_f1, "acc": val_acc, "mcc": val_mcc}
        )

        print(
            f"[{tag}] Ep{epoch}: trL={tr_loss:.3f} valL={val_loss:.3f} "
            f"trF1={tr_f1:.3f} valF1={val_f1:.3f} valACC={val_acc:.3f} valMCC={val_mcc:.3f} "
            f"({time.time()-t0:.1f}s)"
        )

    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts


# ------------------------------------------------------------------ #
# execute training runs
train(BaseTransformer(use_positional=True), tag="Positional_Baseline")
train(BaseTransformer(use_positional=False), tag="NoPos_Baseline")

# ------------------------------------------------------------------ #
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
