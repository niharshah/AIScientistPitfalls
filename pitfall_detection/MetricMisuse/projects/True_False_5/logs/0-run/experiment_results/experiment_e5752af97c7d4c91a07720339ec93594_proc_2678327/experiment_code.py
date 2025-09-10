import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -------------------- workspace & device --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
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


# -------------------- helper metrics --------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / (sum(w) if sum(w) else 1.0)


# -------------------- load SPR_BENCH --------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # read one csv as a split
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# -------------------- build vocabularies --------------------
token_vocab = {"<pad>": 0, "<unk>": 1}
shape_vocab, color_vocab = {}, {}


def register_token(tok):
    if tok not in token_vocab:
        token_vocab[tok] = len(token_vocab)
    if tok:
        shape = tok[0]
        if shape not in shape_vocab:
            shape_vocab[shape] = len(shape_vocab)
        if len(tok) > 1:
            color = tok[1]
            if color not in color_vocab:
                color_vocab[color] = len(color_vocab)


for seq in spr["train"]["sequence"]:
    for t in seq.strip().split():
        register_token(t)

print(
    f"Token vocab: {len(token_vocab)}, Shapes: {len(shape_vocab)}, Colors: {len(color_vocab)}"
)


# -------------------- dataset --------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def _encode_tokens(self, seq):
        return [
            token_vocab.get(tok, token_vocab["<unk>"]) for tok in seq.strip().split()
        ]

    def _shape_color_counts(self, seq):
        s_vec = np.zeros(len(shape_vocab), dtype=np.float32)
        c_vec = np.zeros(len(color_vocab), dtype=np.float32)
        for tok in seq.strip().split():
            if not tok:
                continue
            s_vec[shape_vocab[tok[0]]] += 1.0
            if len(tok) > 1:
                c_vec[color_vocab[tok[1]]] += 1.0
        return np.concatenate([s_vec, c_vec])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(self._encode_tokens(seq), dtype=torch.long),
            "sym_feats": torch.tensor(
                self._shape_color_counts(seq), dtype=torch.float32
            ),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "seq_str": seq,
        }


train_ds = SPRDataset(spr["train"])
dev_ds = SPRDataset(spr["dev"])
test_ds = SPRDataset(spr["test"])

# -------------------- collate --------------------
PAD_IDX = token_vocab["<pad>"]


def collate(batch):
    ids = [b["input_ids"] for b in batch]
    feats = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    seqs = [b["seq_str"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD_IDX)
    mask = (padded != PAD_IDX).float()
    return {
        "input_ids": padded,
        "mask": mask,
        "sym_feats": feats,
        "labels": labels,
        "seq_str": seqs,
    }


BATCH = 256
train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, BATCH, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate)


# -------------------- model --------------------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, sym_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + sym_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, ids, mask, sym_feats):
        emb = self.emb(ids) * mask.unsqueeze(-1)  # B,T,D
        mean_emb = emb.sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        out = torch.cat([mean_emb, sym_feats], dim=-1)
        return self.fc(out)


num_classes = int(max(train_ds.labels)) + 1
model = HybridClassifier(
    len(token_vocab), 64, len(shape_vocab) + len(color_vocab), num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


# -------------------- evaluation --------------------
def run_epoch(loader, train=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"], batch["mask"], batch["sym_feats"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        pred = logits.argmax(-1).cpu().tolist()
        gt = batch["labels"].cpu().tolist()
        preds.extend(pred)
        gts.extend(gt)
        seqs.extend(batch["seq_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return total_loss / len(loader.dataset), swa, preds, gts, seqs


# -------------------- training loop --------------------
EPOCHS, PATIENCE = 20, 3
best_val_swa, patience_cnt, best_state = -1.0, 0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_swa, *_ = run_epoch(train_loader, train=True)
    val_loss, val_swa, *_ = run_epoch(dev_loader, train=False)

    exp_rec["losses"]["train"].append(tr_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(tr_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_SWA={val_swa:.4f}"
    )

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping (no val_SWA improvement for {PATIENCE} epochs)")
            break

# -------------------- test evaluation --------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, _ = run_epoch(test_loader, train=False)
print(f"TEST: loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# -------------------- save --------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
