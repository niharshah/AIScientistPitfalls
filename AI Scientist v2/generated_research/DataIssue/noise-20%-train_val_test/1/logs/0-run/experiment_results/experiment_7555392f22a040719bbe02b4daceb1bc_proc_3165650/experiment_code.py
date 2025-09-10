import os, pathlib, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------------------------#
# working dir & device                                                         #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment data                                                              #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_macroF1": [], "val_macroF1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------------------------#
# locate dataset                                                               #
def find_spr_bench():
    opts = [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]
    for p in opts:
        if p and p.exists() and (p / "train.csv").exists():
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH folder not found")


data_root = find_spr_bench()


def load_spr(root):
    def _ld(split):
        return load_dataset(
            "csv", data_files=str(root / split), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


spr = load_spr(data_root)


# -----------------------------------------------------------------------------#
# vocab                                                                         #
def build_vocab(ds):
    vocab = {"<pad>": 0, "<cls>": 1}
    for ex in ds:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


# -----------------------------------------------------------------------------#
# dataset                                                                       #
class SPRDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.data = hf_ds
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.cls_id = vocab["<cls>"]

    def encode(self, seq):
        return [self.cls_id] + [self.vocab[t] for t in seq.split()]

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(self.encode(row["sequence"]), dtype=torch.long)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        # auxiliary labels
        length_parity = torch.tensor(len(ids) % 2, dtype=torch.long)
        uniq = len(set(row["sequence"].split()))
        uniq_clip = min(uniq, 10)  # 0-10 bucket
        uniq_label = torch.tensor(uniq_clip, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": label,
            "parity": length_parity,
            "uniq": uniq_label,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(batch, pad_id):
    keys = ["input_ids", "labels", "parity", "uniq"]
    col = {k: [b[k] for b in batch] for k in keys}
    seqs = nn.utils.rnn.pad_sequence(
        col["input_ids"], batch_first=True, padding_value=pad_id
    )
    attn = (seqs != pad_id).long()
    return {
        "input_ids": seqs,
        "attention_mask": attn,
        "labels": torch.stack(col["labels"]),
        "parity": torch.stack(col["parity"]),
        "uniq": torch.stack(col["uniq"]),
    }


train_ds, dev_ds, test_ds = (
    SPRDataset(spr[s], vocab) for s in ["train", "dev", "test"]
)
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)

num_labels = len({int(x["label"]) for x in spr["train"]})
max_len = max(len(x["sequence"].split()) + 1 for x in spr["train"])
print("Num classes:", num_labels, "| max len:", max_len)


# -----------------------------------------------------------------------------#
# model                                                                         #
class PosEnc(nn.Module):
    def __init__(self, d, mlen):
        super().__init__()
        pe = torch.zeros(mlen, d)
        pos = torch.arange(0, mlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(
        self, vocab, classes, aux_par=2, aux_uniq=11, d=192, heads=8, layers=4, drop=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = PosEnc(d, max_len + 5)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(d)
        self.main_head = nn.Linear(d, classes)
        self.par_head = nn.Linear(d, aux_par)
        self.uniq_head = nn.Linear(d, aux_uniq)

    def forward(self, ids, mask):
        x = self.embed(ids)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=(mask == 0))
        cls = self.norm(x[:, 0])
        return self.main_head(cls), self.par_head(cls), self.uniq_head(cls)


model = SPRTransformer(len(vocab), num_labels).to(device)

# -----------------------------------------------------------------------------#
# optimizer / scheduler                                                         #
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion_main = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_aux = nn.CrossEntropyLoss()


# -----------------------------------------------------------------------------#
# metrics                                                                       #
def macro_f1(preds, labels, ncls):
    p, l = preds.cpu().numpy(), labels.cpu().numpy()
    f1s = []
    for c in range(ncls):
        tp = ((p == c) & (l == c)).sum()
        fp = ((p == c) & (l != c)).sum()
        fn = ((p != c) & (l == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        f1s.append(f1)
    return float(np.mean(f1s))


# -----------------------------------------------------------------------------#
# train / eval loop                                                             #
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, preds_all, labels_all = 0, [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        main_logits, par_logits, uniq_logits = model(
            batch["input_ids"], batch["attention_mask"]
        )
        loss = (
            criterion_main(main_logits, batch["labels"])
            + 0.3 * criterion_aux(par_logits, batch["parity"])
            + 0.3 * criterion_aux(uniq_logits, batch["uniq"])
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        with torch.no_grad():
            preds_all.append(main_logits.argmax(1))
            labels_all.append(batch["labels"])
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    macro = macro_f1(preds_all, labels_all, num_labels)
    return tot_loss / len(loader.dataset), macro


# -----------------------------------------------------------------------------#
# training loop with early stopping                                            #
best_macro, patience, max_pat = 0, 0, 3
epochs = 15
for epoch in range(1, epochs + 1):
    tr_loss, tr_macro = run_epoch(train_loader, True)
    val_loss, val_macro = run_epoch(dev_loader, False)
    sched.step()
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_macroF1"].append(tr_macro)
    experiment_data["SPR_BENCH"]["metrics"]["val_macroF1"].append(val_macro)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_macroF1 = {val_macro:.4f}"
    )
    if val_macro > best_macro:
        best_macro = val_macro
        patience = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best.pt"))
    else:
        patience += 1
        if patience > max_pat:
            print("Early stopping")
            break

# -----------------------------------------------------------------------------#
# load best & test                                                             #
model.load_state_dict(torch.load(os.path.join(working_dir, "best.pt")))
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        main, _, _ = model(batch["input_ids"], batch["attention_mask"])
        preds_all.append(main.argmax(1).cpu())
        labels_all.append(batch["labels"].cpu())
preds_all = torch.cat(preds_all)
labels_all = torch.cat(labels_all)
test_macro = macro_f1(preds_all, labels_all, num_labels)
test_acc = (preds_all == labels_all).float().mean().item()
print(f"Test accuracy: {test_acc*100:.2f}% | Test macroF1: {test_macro:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds_all.numpy()
experiment_data["SPR_BENCH"]["ground_truth"] = labels_all.numpy()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data")
