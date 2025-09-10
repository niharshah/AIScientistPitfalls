import os, time, pathlib, random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data dict ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
exp_rec = experiment_data["SPR_BENCH"]


# ---------- util funcs ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ---------- dataset path ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab & maps ----------
def build_vocab_and_maps(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    shapes, colors = set(), set()
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
            shapes.add(tok[0])
            if len(tok) > 1:
                colors.add(tok[1])
    shape_map = {s: i for i, s in enumerate(sorted(shapes))}
    color_map = {c: i for i, c in enumerate(sorted(colors))}
    return vocab, shape_map, color_map


vocab, shape_map, color_map = build_vocab_and_maps(spr["train"])
sym_dim = len(shape_map) + len(color_map)
print(f"vocab={len(vocab)} shapes={len(shape_map)} colors={len(color_map)}")


# ---------- torch Dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, shape_map, color_map):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.vocab = vocab
        self.shape_map = shape_map
        self.color_map = color_map
        self.sym_len = len(shape_map) + len(color_map)

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return [self.vocab.get(tok, 1) for tok in seq.strip().split()]

    def sym_vec(self, seq):
        s = np.zeros(len(self.shape_map), dtype=np.float32)
        c = np.zeros(len(self.color_map), dtype=np.float32)
        for tok in seq.strip().split():
            s[self.shape_map[tok[0]]] += 1
            if len(tok) > 1:
                c[self.color_map[tok[1]]] += 1
        tot = max(1, len(seq.strip().split()))
        return np.concatenate([s, c]) / tot

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(self.encode(seq), dtype=torch.long),
            "sym": torch.tensor(self.sym_vec(seq), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_str": seq,
        }


train_ds = SPRTorchDataset(spr["train"], vocab, shape_map, color_map)
dev_ds = SPRTorchDataset(spr["dev"], vocab, shape_map, color_map)
test_ds = SPRTorchDataset(spr["test"], vocab, shape_map, color_map)


# ---------- collate ----------
def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    sym = torch.stack([b["sym"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    return {
        "input_ids": padded,
        "mask": mask,
        "sym": sym,
        "labels": labels,
        "seqs": seqs,
    }


BATCH = 128
train_loader = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn
)


# ---------- model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # 1 x L x D

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class NeuralSymbolicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        sym_dim,
        n_heads,
        n_layers,
        hidden_dim,
        n_cls,
        p_drop=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=p_drop,
            batch_first=True,
        )
        self.tr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(p_drop)
        # gating fusion
        self.gate = nn.Linear(emb_dim + sym_dim, emb_dim + sym_dim)
        self.classifier = nn.Linear(emb_dim + sym_dim, n_cls)

    def forward(self, ids, mask, sym, token_dropout=0.0):
        if self.training and token_dropout > 0:
            # randomly drop tokens (set to pad)
            drop_mask = (torch.rand(ids.shape, device=ids.device) < token_dropout) & (
                ids != 0
            )
            ids = ids.masked_fill(drop_mask, 0)
            mask = (ids != 0).float()
        x = self.emb(ids)
        x = self.pos(x)
        x = self.tr_encoder(x, src_key_padding_mask=(mask == 0))
        # mean pool
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1e-6
        )
        fused = torch.cat([pooled, sym], dim=-1)
        g = torch.sigmoid(self.gate(fused))
        fused = fused * g  # gated features
        fused = self.dropout(fused)
        return self.classifier(fused)


num_classes = int(max(train_ds.labels)) + 1
model = NeuralSymbolicTransformer(
    vocab_size=len(vocab),
    emb_dim=64,
    sym_dim=sym_dim,
    n_heads=4,
    n_layers=2,
    hidden_dim=128,
    n_cls=num_classes,
    p_drop=0.1,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


# ---------- evaluation ----------
def evaluate(loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["mask"], batch["sym"])
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            pred = logits.argmax(-1).cpu().tolist()
            preds.extend(pred)
            gts.extend(batch["labels"].cpu().tolist())
            seqs.extend(batch["seqs"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return total_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- training ----------
MAX_EPOCHS, patience = 25, 5
best_val_swa, best_state, no_imp = -1.0, None, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"], batch["mask"], batch["sym"], token_dropout=0.1
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    train_loss_eval, train_swa, *_ = evaluate(train_loader)
    val_loss, val_swa, *_ = evaluate(dev_loader)

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(train_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: validation_loss = {val_loss:.4f}, val_SWA = {val_swa:.4f}"
    )

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break

# ---------- test ----------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, _ = evaluate(test_loader)
print(f"TEST: loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
