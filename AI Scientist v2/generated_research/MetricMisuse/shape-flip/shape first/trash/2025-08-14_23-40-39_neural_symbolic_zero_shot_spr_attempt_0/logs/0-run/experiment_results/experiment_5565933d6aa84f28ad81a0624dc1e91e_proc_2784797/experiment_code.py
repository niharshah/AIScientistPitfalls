# Token-Order-Shuffling Ablation for SPR-BENCH
import os, math, pathlib, numpy as np, torch, random, warnings
from collections import Counter
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------- experiment container -----------------------
experiment_data = {
    "TokenOrderShuffling": {
        "SPR_BENCH": {
            "metrics": {"train_loss": [], "val_loss": [], "val_swa": []},
            "predictions": {"dev": [], "test": []},
            "ground_truth": {"dev": [], "test": []},
            "timestamps": [],
        }
    }
}

# -------------------- working dir / device ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- utils -------------------------------------
PAD, UNK = "<pad>", "<unk>"


def resolve_spr_path() -> pathlib.Path:
    env = os.getenv("SPR_PATH")
    if env and (pathlib.Path(env) / "train.csv").exists():
        return pathlib.Path(env)
    cur = pathlib.Path.cwd()
    for p in [cur] + list(cur.parents):
        if (p / "SPR_BENCH" / "train.csv").exists():
            return p / "SPR_BENCH"
    raise FileNotFoundError("SPR_BENCH dataset not found")


def load_spr(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv",
            data_files=str(root / name),
            split="train",
            cache_dir=str(working_dir) + "/.hf_cache",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1.0)


def shuffle_sequence(seq: str) -> str:
    toks = seq.split()
    random.shuffle(toks)
    return " ".join(toks)


# -------------------- data ---------------------------------------
DATA_PATH = resolve_spr_path()
dsets = load_spr(DATA_PATH)

# vocab build from original (unshuffled) sequences
counter = Counter(tok for seq in dsets["train"]["sequence"] for tok in seq.split())
vocab = {PAD: 0, UNK: 1}
for tok in counter:
    vocab[tok] = len(vocab)
id2tok = {i: t for t, i in vocab.items()}

labels = sorted(set(dsets["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in enumerate(labels)}
num_classes = len(labels)
print(f"Vocabulary size: {len(vocab)} | Classes: {num_classes}")


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq = split["sequence"]
        self.lab = split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        original_seq = self.seq[idx]
        shuffled_seq = shuffle_sequence(original_seq)  # Token-order shuffling
        toks = torch.tensor(encode(shuffled_seq), dtype=torch.long)
        sym_feats = torch.tensor(
            [
                len(shuffled_seq.split()),
                count_shape_variety(shuffled_seq),
                count_color_variety(shuffled_seq),
            ],
            dtype=torch.float32,
        )
        label = torch.tensor(lab2id[self.lab[idx]], dtype=torch.long)
        return {
            "input_ids": toks,
            "sym_feats": sym_feats,
            "label": label,
            "seq_str": shuffled_seq,
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    sym = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {"input_ids": input_ids, "sym_feats": sym, "labels": labels, "seqs": seqs}


BS = 128
train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=BS, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=BS, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(dsets["test"]), batch_size=BS, shuffle=False, collate_fn=collate
)

# -------------------- model --------------------------------------
EMB = 128
SYM_DIM = 3
SYM_PROJ = 32
N_HEAD = 4
N_LAY = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NeuralSymbolicSPR(nn.Module):
    def __init__(self, vocab_sz, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=N_HEAD,
            dim_feedforward=emb_dim * 2,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAY)
        self.sym_proj = nn.Sequential(nn.Linear(SYM_DIM, SYM_PROJ), nn.ReLU())
        self.fc = nn.Linear(emb_dim + SYM_PROJ, num_labels)

    def forward(self, input_ids, sym_feats):
        mask = input_ids == 0
        x = self.emb(input_ids)
        x = self.pos(x)
        h = self.enc(x, src_key_padding_mask=mask)
        seq_emb = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1e-6
        ).unsqueeze(-1)
        sym = self.sym_proj(sym_feats)
        return self.fc(torch.cat([seq_emb, sym], dim=-1))


model = NeuralSymbolicSPR(len(vocab), EMB, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# -------------------- helpers ------------------------------------
def evaluate(loader):
    model.eval()
    tot, n = 0.0, 0
    all_pred, all_lab, all_seq = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["sym_feats"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot += loss.item() * bs
            n += bs
            preds = logits.argmax(1).cpu().tolist()
            labs = batch["labels"].cpu().tolist()
            seqs = batch["seqs"]
            all_pred.extend(preds)
            all_lab.extend(labs)
            all_seq.extend(seqs)
    swa = shape_weighted_accuracy(all_seq, all_lab, all_pred)
    return tot / n, swa, all_pred, all_lab


# -------------------- training -----------------------------------
EPOCHS = 20
PATIENCE = 3
best_swa, patience, best_state = -1, 0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tloss, seen = 0.0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        opt.zero_grad()
        logits = model(batch["input_ids"], batch["sym_feats"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        opt.step()
        bs = batch["labels"].size(0)
        tloss += loss.item() * bs
        seen += bs

    val_loss, val_swa, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch:02d}: train_loss={tloss/seen:.4f} | "
        f"val_loss={val_loss:.4f} | val_SWA={val_swa:.4f}"
    )

    ed = experiment_data["TokenOrderShuffling"]["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(tloss / seen)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_swa"].append(val_swa)
    ed["timestamps"].append(datetime.utcnow().isoformat())

    if val_swa > best_swa:
        best_swa, best_state, patience = val_swa, model.state_dict(), 0
    else:
        patience += 1
    if patience >= PATIENCE:
        print("Early stopping.")
        break

# -------------------- evaluation --------------------------------
if best_state:
    model.load_state_dict(best_state)

dev_loss, dev_swa, dev_pred, dev_lab = evaluate(dev_loader)
test_loss, test_swa, test_pred, test_lab = evaluate(test_loader)

print(f"=== DEV  === loss {dev_loss:.4f} | SWA {dev_swa:.4f}")
print(f"=== TEST === loss {test_loss:.4f} | SWA {test_swa:.4f}")

ed = experiment_data["TokenOrderShuffling"]["SPR_BENCH"]
ed["predictions"]["dev"], ed["ground_truth"]["dev"] = dev_pred, dev_lab
ed["predictions"]["test"], ed["ground_truth"]["test"] = test_pred, test_lab

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
