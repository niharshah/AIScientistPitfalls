import os, pathlib, math, re, time, json, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------
# mandatory working dir & device declarations
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# helper to load SPR benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    dd["train"] = _load("train.csv")
    dd["dev"] = _load("dev.csv")
    dd["test"] = _load("test.csv")
    return dd


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------
# build vocabulary
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

# labels
label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)

# ------------------------------------------------------------
# dataset wrapper
MAX_LEN = 128


def encode_sequence(seq: str):
    tokens = [CLS] + seq.strip().split()
    tok_ids = [vocab.get(t, vocab[UNK]) for t in tokens][:MAX_LEN]
    attn = [1] * len(tok_ids)
    if len(tok_ids) < MAX_LEN:
        pad = MAX_LEN - len(tok_ids)
        tok_ids += [vocab[PAD]] * pad
        attn += [0] * pad
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


train_ds = SPRDataset(spr["train"])
dev_ds = SPRDataset(spr["dev"])

# ------------------------------------------------------------
# dataloaders
BATCH_SIZE = 64


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ------------------------------------------------------------
# model definition factory
class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        dropout_prob,
        d_model=128,
        nhead=4,
        nlayers=2,
        dim_ff=256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        logits = self.classifier(x[:, 0, :])
        return logits


# ------------------------------------------------------------
# training / evaluation helpers
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ------------------------------------------------------------
# hyperparameter tuning loop
EPOCHS = 5
DROP_RATES = [0.0, 0.1, 0.2, 0.3]

experiment_data = {"dropout_probability": {"SPR_BENCH": {}}}

for dr in DROP_RATES:
    print(f"\n=== Training with dropout={dr} ===")
    model = SimpleTransformer(vocab_size, num_labels, dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_pred, val_gt = run_epoch(model, dev_loader, criterion)

        exp_rec["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1})
        exp_rec["metrics"]["val"].append({"epoch": epoch, "macro_f1": val_f1})
        exp_rec["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        exp_rec["losses"]["val"].append({"epoch": epoch, "loss": val_loss})

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_F1={tr_f1:.4f}, val_F1={val_f1:.4f} (time {time.time()-t0:.1f}s)"
        )

    # store final predictions / ground truth
    exp_rec["predictions"] = val_pred
    exp_rec["ground_truth"] = val_gt
    experiment_data["dropout_probability"]["SPR_BENCH"][str(dr)] = exp_rec

    # free gpu memory
    del model
    torch.cuda.empty_cache()

# ------------------------------------------------------------
# save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
