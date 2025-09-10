import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from typing import Dict, List

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


# ---------- helper functions ----------
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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w) if sum(w) else 0.0


# ---------- data path ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- build vocab & symbolic maps ----------
def build_vocab_and_maps(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    shape_set, color_set = set(), set()
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
            if len(tok) >= 1:
                shape_set.add(tok[0])
            if len(tok) >= 2:
                color_set.add(tok[1])
    shape_map = {s: i for i, s in enumerate(sorted(shape_set))}
    color_map = {c: i for i, c in enumerate(sorted(color_set))}
    return vocab, shape_map, color_map


vocab, shape_map, color_map = build_vocab_and_maps(spr["train"])
print(f"Vocab: {len(vocab)}  shapes:{len(shape_map)} colors:{len(color_map)}")


# ---------- torch dataset ----------
class SPRTorchDataset(Dataset):
    def __init__(self, split, vocab, shape_map, color_map):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.vocab = vocab
        self.shape_map = shape_map
        self.color_map = color_map
        self.n_shape = len(shape_map)
        self.n_color = len(color_map)

    def __len__(self):
        return len(self.seqs)

    def encode_tokens(self, seq: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.strip().split()]

    def symbolic_vec(self, seq: str) -> np.ndarray:
        s_vec = np.zeros(self.n_shape, dtype=np.float32)
        c_vec = np.zeros(self.n_color, dtype=np.float32)
        for tok in seq.strip().split():
            if tok and tok[0] in self.shape_map:
                s_vec[self.shape_map[tok[0]]] += 1.0
            if len(tok) > 1 and tok[1] in self.color_map:
                c_vec[self.color_map[tok[1]]] += 1.0
        # normalise by length
        total = max(len(seq.strip().split()), 1)
        return np.concatenate([s_vec, c_vec]) / total

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(self.encode_tokens(seq), dtype=torch.long),
            "sym_feats": torch.tensor(self.symbolic_vec(seq), dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence_str": seq,
        }


train_ds = SPRTorchDataset(spr["train"], vocab, shape_map, color_map)
dev_ds = SPRTorchDataset(spr["dev"], vocab, shape_map, color_map)
test_ds = SPRTorchDataset(spr["test"], vocab, shape_map, color_map)


# ---------- collate ----------
def collate_fn(batch):
    ids = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    mask = (padded != 0).float()
    sym = torch.stack([b["sym_feats"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    return {
        "input_ids": padded,
        "attention_mask": mask,
        "sym_feats": sym,
        "labels": labels,
        "sequence_str": seqs,
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
class NeuralSymbolic(nn.Module):
    def __init__(self, vocab_size, emb_dim, sym_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + sym_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids, mask, sym):
        emb = self.embed(ids)  # B x T x D
        mean_emb = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-6)
        x = torch.cat([mean_emb, sym], dim=-1)
        return self.mlp(x)


num_classes = int(max(train_ds.labels)) + 1
sym_dim = len(shape_map) + len(color_map)
model = NeuralSymbolic(len(vocab), 64, sym_dim, 128, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- eval ----------
def evaluate(loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(
                batch_t["input_ids"], batch_t["attention_mask"], batch_t["sym_feats"]
            )
            loss = criterion(logits, batch_t["labels"])
            total_loss += loss.item() * batch_t["labels"].size(0)
            pred = logits.argmax(dim=-1).cpu().tolist()
            gt = batch_t["labels"].cpu().tolist()
            preds.extend(pred)
            gts.extend(gt)
            seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return total_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- training ----------
MAX_EPOCHS, patience = 25, 5
best_val_swa, best_state, epochs_no_imp = -1.0, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(
            batch_t["input_ids"], batch_t["attention_mask"], batch_t["sym_feats"]
        )
        loss = criterion(logits, batch_t["labels"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_t["labels"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    train_loss_eval, train_swa, *_ = evaluate(train_loader)
    val_loss, val_swa, *_ = evaluate(dev_loader)

    exp_rec["losses"]["train"].append(train_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_swa"].append(train_swa)
    exp_rec["metrics"]["val_swa"].append(val_swa)
    exp_rec["timestamps"].append(time.time())

    print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  val_SWA={val_swa:.4f}")

    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        epochs_no_imp = 0
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

# ---------- test ----------
model.load_state_dict(best_state)
test_loss, test_swa, test_preds, test_gts, test_seqs = evaluate(test_loader)
print(f"TEST   loss={test_loss:.4f}  SWA={test_swa:.4f}")

exp_rec["predictions"] = np.array(test_preds)
exp_rec["ground_truth"] = np.array(test_gts)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
