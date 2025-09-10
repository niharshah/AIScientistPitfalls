import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# ---- working dir & device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---- helper metrics ---------------------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) else 1.0)


# ---- load SPR_BENCH ---------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
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


# ---- vocabulary -------------------------------------------------------------
def build_vocab(hf_dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in hf_dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


# ---- Torch Dataset ----------------------------------------------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def encode(self, seq: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.strip().split()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_str = self.seqs[idx]
        return {
            "input_ids": torch.tensor(self.encode(seq_str), dtype=torch.long),
            "sym_counts": torch.tensor(
                [count_shape_variety(seq_str), count_color_variety(seq_str)],
                dtype=torch.float,
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": seq_str,
        }


train_ds, dev_ds, test_ds = [
    SPRTorchDataset(spr[s], vocab) for s in ("train", "dev", "test")
]


# ---- collate ---------------------------------------------------------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    counts = torch.stack([b["sym_counts"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "sym_counts": counts,
        "labels": labels,
        "sequence_str": seqs,
    }


BATCH = 128
train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, BATCH, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate)


# ---- model -----------------------------------------------------------------
class NeuralSymbolicClassifier(nn.Module):
    def __init__(
        self, vocab_size, num_classes, emb_dim=128, nhead=4, depth=2, dim_feed=256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_feed, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.count_proj = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.classifier = nn.Linear(emb_dim + 32, num_classes)

    def forward(self, ids, mask, counts):
        x = self.embed(ids) * mask.unsqueeze(-1)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1e-6)
        c_feat = self.count_proj(counts)
        logits = self.classifier(torch.cat([pooled, c_feat], dim=-1))
        return logits


num_classes = int(max(train_ds.labels)) + 1
model = NeuralSymbolicClassifier(len(vocab), num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# ---- experiment data dict ---------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
rec = experiment_data["SPR_BENCH"]


# ---- evaluation -------------------------------------------------------------
def evaluate(loader):
    model.eval()
    tot_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(bt["input_ids"], bt["attention_mask"], bt["sym_counts"])
            loss = criterion(logits, bt["labels"])
            tot_loss += loss.item() * bt["labels"].size(0)
            pred = logits.argmax(-1).cpu().tolist()
            gt = bt["labels"].cpu().tolist()
            preds.extend(pred)
            gts.extend(gt)
            seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return tot_loss / len(loader.dataset), swa, preds, gts, seqs


# ---- training loop with early stopping -------------------------------------
MAX_EPOCHS = 20
patience = 3
best_val_swa = -1
wait = 0
best_state = None
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for batch in train_loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(bt["input_ids"], bt["attention_mask"], bt["sym_counts"])
        loss = criterion(logits, bt["labels"])
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * bt["labels"].size(0)
    train_loss = ep_loss / len(train_loader.dataset)
    val_loss, val_swa, _, _, _ = evaluate(dev_loader)
    train_loss_eval, train_swa, _, _, _ = evaluate(train_loader)

    rec["losses"]["train"].append(train_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_swa"].append(train_swa)
    rec["metrics"]["val_swa"].append(val_swa)
    rec["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_SWA = {val_swa:.4f}")

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ---- final test evaluation --------------------------------------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, test_seqs = evaluate(test_loader)
print(f"TEST: loss={test_loss:.4f}  SWA={test_swa:.4f}")

rec["predictions"] = np.array(test_preds)
rec["ground_truth"] = np.array(test_gts)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
