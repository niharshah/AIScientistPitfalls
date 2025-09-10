import os, pathlib, time, random, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# reproducibility
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------
# experiment data structure
experiment_data = {
    "SPR_BENCH": {
        "baseline": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "symbolic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# ------------------------------------------------------------------
# load SPR benchmark
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# simple whitespace vocab
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add_tok(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add_tok(tok)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: lab for lab, i in label2id.items()}
num_labels = len(label2id)
print("Num labels:", num_labels)

# ------------------------------------------------------------------
# dataset / dataloader
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


BATCH = 64


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_loader = DataLoader(
    SPRDataset(spr["train"]), BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), BATCH, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), BATCH, shuffle=False, collate_fn=collate_fn
)


# ------------------------------------------------------------------
# models
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=2, d_ff=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_lay = nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_lay, nlayers)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        logits = self.cls(x[:, 0, :])
        return logits


class SymbolicFeatureExtractor(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, out_dim), nn.ReLU()
        )

    def forward(self, input_ids, attention_mask):
        inp = input_ids.detach().cpu()
        att = attention_mask.detach().cpu()
        feats = []
        for ids, mask in zip(inp, att):
            toks = ids[mask.bool()]
            seq_len = toks.size(0)
            uniq = len(torch.unique(toks))
            uniq_ratio = uniq / seq_len if seq_len > 0 else 0.0
            feats.append([seq_len / MAX_LEN, uniq / MAX_LEN, uniq_ratio])
        feats = torch.tensor(feats, dtype=torch.float, device=input_ids.device)
        return self.mlp(feats)


class SymbolicAugTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=2, d_ff=256, sym_dim=32):
        super().__init__()
        self.trans = SimpleTransformer(d_model, nhead, nlayers, d_ff)
        self.sym = SymbolicFeatureExtractor(sym_dim)
        self.cls = nn.Linear(d_model + sym_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        # get transformer hidden
        x = self.trans.embed(input_ids) + self.trans.pos[:, : input_ids.size(1), :]
        x = self.trans.encoder(x, src_key_padding_mask=~attention_mask.bool())
        cls_rep = x[:, 0, :]
        sym_rep = self.sym(input_ids, attention_mask)
        logits = self.cls(torch.cat([cls_rep, sym_rep], dim=-1))
        return logits


# ------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)
    return tot_loss / len(loader.dataset), f1, acc, preds, gts


# ------------------------------------------------------------------
def train_model(model, tag, epochs=3, lr=3e-4):
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        s = time.time()
        tr_loss, tr_f1, tr_acc, _, _ = run_epoch(model, train_loader, optim)
        val_loss, val_f1, val_acc, val_pred, val_gt = run_epoch(model, dev_loader)
        d = experiment_data["SPR_BENCH"][tag]
        d["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1, "acc": tr_acc})
        d["metrics"]["val"].append({"epoch": epoch, "macro_f1": val_f1, "acc": val_acc})
        d["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        d["losses"]["val"].append({"epoch": epoch, "loss": val_loss})
        print(
            f"{tag} | Ep {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f} ({time.time()-s:.1f}s)"
        )
    # store final predictions
    d["predictions"] = val_pred
    d["ground_truth"] = val_gt
    return model


# ------------------------------------------------------------------
print("\n=== Baseline ===")
baseline_model = train_model(
    SimpleTransformer(d_model=128, nhead=4), "baseline", epochs=3
)

print("\n=== Symbolic Augmented ===")
sym_model = train_model(
    SymbolicAugTransformer(d_model=128, nhead=4), "symbolic", epochs=3
)

# ------------------------------------------------------------------
# final evaluation on test set for best model (symbolic)
test_loss, test_f1, test_acc, test_pred, test_gt = run_epoch(sym_model, test_loader)
print(
    f"\nSymbolic model TEST: loss={test_loss:.4f}  F1={test_f1:.4f}  ACC={test_acc:.4f}"
)
experiment_data["SPR_BENCH"]["symbolic"]["test_metrics"] = {
    "loss": test_loss,
    "macro_f1": test_f1,
    "acc": test_acc,
}

# save everything
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
