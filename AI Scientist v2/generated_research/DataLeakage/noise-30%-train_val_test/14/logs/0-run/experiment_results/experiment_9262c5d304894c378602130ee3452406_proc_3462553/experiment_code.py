import os, pathlib, time, random, math, re, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------
# small util to load benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for split in ["train", "dev", "test"]:
        dd[split] = _load(f"{split}.csv")
    return dd


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------------------------------------------------------------
# vocabulary
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add_token(t):
    if t not in vocab:
        vocab[t] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add_token(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)

# ---------------------------------------------------------------------
# dataset & dataloader helpers
MAX_LEN = 128


def encode_sequence(seq: str):
    toks = [CLS] + seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


BATCH_SIZE = 64
train_loader_base = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader_base = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# ---------------------------------------------------------------------
# model definition
class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=128, nhead=4, nlayers=2, dim_ff=256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn):
        x = self.embed(ids) + self.pos[:, : ids.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~attn.bool())
        cls = x[:, 0]
        return self.classifier(cls)


# ---------------------------------------------------------------------
# training helpers
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
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
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------------------------------------------------------------
# hyper-parameter tuning over different epoch budgets
epoch_options = [5, 10, 15, 20]
experiment_data = {"epochs_tuning": {}}

for EPOCHS in epoch_options:
    print(f"\n===== Training for {EPOCHS} epochs =====")
    # reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    # fresh model & optimiser
    model = SimpleTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # copy loaders (they are small wrappers, reuse OK)
    train_loader, dev_loader = train_loader_base, dev_loader_base

    run_record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_f1, val_pred, val_gt = run_epoch(model, dev_loader)
        run_record["metrics"]["train"].append({"epoch": ep, "macro_f1": tr_f1})
        run_record["metrics"]["val"].append({"epoch": ep, "macro_f1": val_f1})
        run_record["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        run_record["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        print(
            f"Ep {ep}/{EPOCHS}  tr_loss:{tr_loss:.4f} val_loss:{val_loss:.4f}  tr_F1:{tr_f1:.4f} val_F1:{val_f1:.4f}  ({time.time()-t0:.1f}s)"
        )
    run_record["predictions"] = val_pred
    run_record["ground_truth"] = val_gt
    # store under experiment_data
    experiment_data["epochs_tuning"][f"E{EPOCHS}"] = {"SPR_BENCH": run_record}

# ---------------------------------------------------------------------
# save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy to", working_dir)
