import os, pathlib, random, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# --------------------------------- boilerplate ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

experiment_data = {
    "SPR_BENCH": {
        "Baseline": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "SymbolicAug": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# ---------------------------- data & vocabulary --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_file):
        return load_dataset(
            "csv",
            data_files=str(root / split_file),
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

CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add(tok)
vocab_size = len(vocab)
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"Vocab:{vocab_size}  Labels:{num_labels}")

MAX_LEN = 128


def encode(seq: str):
    toks = [CLS] + seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf):
        self.seqs, self.labels = hf["sequence"], hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(label2id[self.labels[idx]]),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


BATCH = 64
train_loader = DataLoader(SPRDataset(spr["train"]), BATCH, True, collate_fn=collate)
dev_loader = DataLoader(SPRDataset(spr["dev"]), BATCH, False, collate_fn=collate)


# ---------------------------- model definitions --------------------------------
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_l = nn.TransformerEncoderLayer(d_model, n_head, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_l, n_layer)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, mask):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~mask.bool())
        return self.cls(x[:, 0])


class SymbolicAugTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, n_layer=2, ff=256, sym_dim=64):
        super().__init__()
        self.baseline = BaselineTransformer(d_model, n_head, n_layer, ff)
        # symbolic path
        self.sym_mlp = nn.Sequential(
            nn.Linear(vocab_size, sym_dim), nn.ReLU(), nn.LayerNorm(sym_dim)
        )
        self.final = nn.Linear(d_model + sym_dim, num_labels)

    def forward(self, ids, mask):
        # transformer branch
        x = self.baseline.emb(ids) + self.baseline.pos[:, : ids.size(1)]
        x = self.baseline.enc(x, src_key_padding_mask=~mask.bool())
        h_token = x[:, 0]  # (B,d_model)
        # symbolic count branch
        counts = []
        for seq in ids:  # loop over batch
            c = torch.bincount(seq, minlength=vocab_size)
            counts.append(c)
        counts = torch.stack(counts).float().to(device)  # (B,V)
        h_sym = self.sym_mlp(counts)
        h = torch.cat([h_token, h_sym], dim=-1)
        return self.final(h)


# ------------------------- training / evaluation utils -------------------------
crit = nn.CrossEntropyLoss()


def loop(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = crit(logits, batch["labels"])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)  # used as RGA proxy
    return tot_loss / len(loader.dataset), macro_f1, acc, preds, gts


# --------------------------------- run exp -------------------------------------
def train_model(name, model, epochs=3, lr=3e-4):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _, _ = loop(model, train_loader, optim)
        val_loss, val_f1, val_acc, pred, gt = loop(model, dev_loader)
        ed = experiment_data["SPR_BENCH"][name]
        ed["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1, "RGA": None})
        ed["metrics"]["val"].append(
            {"epoch": epoch, "macro_f1": val_f1, "RGA": val_acc}
        )
        ed["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        ed["losses"]["val"].append({"epoch": epoch, "loss": val_loss})
        print(
            f"{name} Epoch {epoch}: "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"trainF1={tr_f1:.3f} valF1={val_f1:.3f} RGA={val_acc:.3f} "
            f"({time.time()-t0:.1f}s)"
        )
    ed["predictions"], ed["ground_truth"] = pred, gt


train_model("Baseline", BaselineTransformer())
train_model("SymbolicAug", SymbolicAugTransformer())

# --------------------------------- save results --------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data.")
