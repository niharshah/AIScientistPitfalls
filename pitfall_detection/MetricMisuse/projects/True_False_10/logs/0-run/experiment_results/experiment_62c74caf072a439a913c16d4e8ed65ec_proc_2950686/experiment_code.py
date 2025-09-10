# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, time, copy, math, warnings, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---- working dir & device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- reproducibility --------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")


# ---- metric: Shape-Weighted Accuracy ---------------------------------------
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---- load / fallback dataset ------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
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


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        ds = load_spr_bench(DATA_PATH)
        print("Loaded SPR_BENCH.")
    except Exception as e:
        print("Dataset not found, generating synthetic data.", e)
        shapes, colors = "ABCD", "abcd"

        def make(n):
            data = [
                {
                    "id": i,
                    "sequence": " ".join(
                        random.choice(shapes) + random.choice(colors)
                        for _ in range(random.randint(3, 10))
                    ),
                    "label": random.choice(["yes", "no"]),
                }
                for i in range(n)
            ]
            return load_dataset("json", data_files={"train": data}, split="train")

        ds = DatasetDict({"train": make(2000), "dev": make(400), "test": make(400)})
    return ds


spr = get_dataset()

# ---- vocabulary & label maps ------------------------------------------------
tokens = {tok for ex in spr["train"] for tok in ex["sequence"].split()}
tok2id = {t: i + 1 for i, t in enumerate(sorted(tokens))}  # 0 reserved for PAD
label2id = {l: i for i, l in enumerate(sorted({ex["label"] for ex in spr["train"]}))}
id2label = {v: k for k, v in label2id.items()}
VOCAB_SIZE = len(tok2id) + 1
NUM_CLS = len(label2id)


# ---- torch dataset ----------------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.lbl = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        token_ids = [tok2id[t] for t in seq.split()]
        n_shape = _count_shape(seq)
        n_color = len(set(tok[1] for tok in seq.split() if len(tok) > 1))
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(len(token_ids)),
            "sym": torch.tensor(
                [n_shape, n_color, n_shape * n_color], dtype=torch.float
            ),
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "raw_seq": seq,
        }


def collate(batch):
    max_len = max(b["length"] for b in batch).item()
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        ids[i, : b["length"]] = b["input_ids"]
        mask[i, : b["length"]] = 1
    return {
        "input_ids": ids,
        "mask": mask,
        "sym": torch.stack([b["sym"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(spr["test"]), batch_size, shuffle=False, collate_fn=collate
)


# ---- model ------------------------------------------------------------------
class NeuroSymbolicTransformer(nn.Module):
    def __init__(self, vocab, embed_dim=64, nhead=4, nlayers=2, sym_dim=3, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, dim_feedforward=128, batch_first=True
        )
        self.trans = nn.TransformerEncoder(encoder_layer, nlayers)
        self.cls = nn.Linear(embed_dim + sym_dim, n_cls)

    def forward(self, ids, mask, sym_feats):
        x = self.emb(ids)
        x = self.trans(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(
            min=1
        )
        return self.cls(torch.cat([pooled, sym_feats], -1))


# ---- helpers ----------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, n_items = 0, 0
    all_true, all_pred, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["mask"], batch["sym"])
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            n_items += batch["label"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            all_true.extend(batch["label"].cpu().tolist())
            all_pred.extend(preds)
            all_seqs.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(all_seqs, all_true, all_pred)
    return total_loss / n_items, swa, all_true, all_pred, all_seqs


# ---- training loop ----------------------------------------------------------
max_epochs = 20
patience = 4
model = NeuroSymbolicTransformer(VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

best_val, wait, best_state = math.inf, 0, None
for epoch in range(1, max_epochs + 1):
    model.train()
    ep_loss, m = 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["mask"], batch["sym"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * batch["label"].size(0)
        m += batch["label"].size(0)
    train_loss = ep_loss / m
    val_loss, val_swa, y_true, y_pred, _ = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA = {val_swa:.4f}")
    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_swa)
    experiment_data["SPR_BENCH"]["predictions"].append(y_pred)
    experiment_data["SPR_BENCH"]["ground_truth"].append(y_true)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    # early stopping
    if val_loss < best_val - 1e-4:
        best_val, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping triggered.")
        break

# ---- test evaluation --------------------------------------------------------
model.load_state_dict(best_state)
test_loss, test_swa, y_tst, y_pst, seqs = evaluate(model, test_loader)
print(f"TEST: loss={test_loss:.4f}, SWA={test_swa:.4f}")

# ---- save results -----------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
