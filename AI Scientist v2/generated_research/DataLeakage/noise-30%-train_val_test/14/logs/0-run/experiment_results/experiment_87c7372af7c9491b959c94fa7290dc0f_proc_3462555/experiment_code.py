import os, pathlib, time, json, random, math, re, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------------------
# mandatory working dir & device declarations
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------------
# util to load benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


# ---------------------------------------------------------------------------------
# 1) Load dataset
SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------------------
# 2) Build vocabulary
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add_token(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add_token(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

# label mapping
label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)

# ---------------------------------------------------------------------------------
# 3) Dataset wrapper
MAX_LEN = 128


def encode_sequence(seq: str):
    tokens = [CLS] + seq.strip().split()
    tok_ids = [vocab.get(t, vocab[UNK]) for t in tokens][:MAX_LEN]
    attn = [1] * len(tok_ids)
    if len(tok_ids) < MAX_LEN:
        pad_len = MAX_LEN - len(tok_ids)
        tok_ids += [vocab[PAD]] * pad_len
        attn += [0] * pad_len
    return tok_ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


train_ds, dev_ds = SPRDataset(spr["train"]), SPRDataset(spr["dev"])


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------------
# 5) Model definition
class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=128, nhead=4, nlayers=2, dim_ff=256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        logits = self.classifier(x[:, 0, :])
        return logits


# ---------------------------------------------------------------------------------
# helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, gts, preds = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------------------------------------------------------------------------
# 7) Hyper-parameter tuning loop (batch size)
EPOCHS = 5
BATCH_SIZES = [32, 64, 128]  # candidate batch sizes

experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}

for bs in BATCH_SIZES:
    print(f"\n====== Training with batch size {bs} ======")
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    model = SimpleTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # prepare storage
    experiment_data["batch_size_tuning"]["SPR_BENCH"][bs] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, optimizer)
        dv_loss, dv_f1, dv_pred, dv_gt = run_epoch(model, dev_loader)

        exp_entry = experiment_data["batch_size_tuning"]["SPR_BENCH"][bs]
        exp_entry["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1})
        exp_entry["metrics"]["val"].append({"epoch": epoch, "macro_f1": dv_f1})
        exp_entry["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        exp_entry["losses"]["val"].append({"epoch": epoch, "loss": dv_loss})

        print(
            f"Epoch {epoch}: "
            f"train_loss={tr_loss:.4f}, val_loss={dv_loss:.4f}, "
            f"train_F1={tr_f1:.4f}, val_F1={dv_f1:.4f}  "
            f"(time {time.time()-t0:.1f}s)"
        )

    # store final preds/gts
    exp_entry["predictions"], exp_entry["ground_truth"] = dv_pred, dv_gt

# ---------------------------------------------------------------------------------
# 8) Save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
