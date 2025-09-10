import os, random, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir & device ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# experiment data container -----------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------------------------------------------------------
# helper to load SPR_BENCH ------------------------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------------------------------------------------------
# build vocabulary ---------------------------------------------------
special_tokens = ["<PAD>", "<MASK>"]
chars = set(ch for seq in spr["train"]["sequence"] for ch in seq)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, mask_id = stoi["<PAD>"], stoi["<MASK>"]
vocab_size = len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"vocab={vocab_size}  classes={num_classes}")


# ------------------------------------------------------------------
# dataset definitions ------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def cls_collate(batch):
    pad = pad_id
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad)
    return {"input_ids": padded, "label": labels}


class MLMDataset(Dataset):
    """
    Produce (input_ids, labels_mlm) where labels_mlm=-100 for non-masked positions.
    """

    def __init__(self, sequences, mlm_prob=0.15):
        self.seqs = sequences
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [stoi[c] for c in self.seqs[idx]]
        ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        # choose mask positions
        mask_positions = torch.bernoulli(torch.full(ids.shape, self.mlm_prob)).bool()
        labels[mask_positions] = ids[mask_positions]
        ids_masked = ids.clone()
        ids_masked[mask_positions] = mask_id
        return {"input_ids": ids_masked, "labels_mlm": labels}


def mlm_collate(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = [b["labels_mlm"] for b in batch]
    seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": seqs_pad, "labels_mlm": labels_pad}


# ------------------------------------------------------------------
# DataLoaders --------------------------------------------------------
train_loader_cls = DataLoader(
    SPRDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=cls_collate
)
dev_loader_cls = DataLoader(
    SPRDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=cls_collate
)
test_loader_cls = DataLoader(
    SPRDataset(spr["test"]), batch_size=128, shuffle=False, collate_fn=cls_collate
)

# MLM uses train+dev sequences (no labels)
mlm_sequences = spr["train"]["sequence"] + spr["dev"]["sequence"]
mlm_loader = DataLoader(
    MLMDataset(mlm_sequences), batch_size=128, shuffle=True, collate_fn=mlm_collate
)


# ------------------------------------------------------------------
# Model --------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=8, nlayers=4, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.randn(512, d_model))  # max len 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, x, pad_mask):
        # x: (B, L)
        L = x.size(1)
        pos_emb = self.pos[:L].unsqueeze(0).expand(x.size(0), -1, -1)
        h = self.embed(x) + pos_emb
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return h  # (B, L, D)


class SPRModel(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.cls_head = nn.Linear(encoder.embed.embedding_dim, n_cls)

    def forward(self, x, pad_mask):
        h = self.encoder(x, pad_mask)  # (B,L,D)
        mask = (~pad_mask).unsqueeze(-1).type_as(h)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.cls_head(pooled)


encoder = Encoder(vocab_size, pad_idx=pad_id).to(device)
lm_head = nn.Linear(encoder.embed.embedding_dim, vocab_size).to(device)
spr_model = SPRModel(encoder, num_classes).to(device)

# ------------------------------------------------------------------
# Pre-training (MLM) -------------------------------------------------
print("=== MLM pre-training ===")
optim_mlm = torch.optim.Adam(
    list(encoder.parameters()) + list(lm_head.parameters()), lr=1e-3
)
loss_mlm_fn = nn.CrossEntropyLoss(ignore_index=-100)

mlm_epochs = 3  # small for demo; can increase if time allows
encoder.train()
lm_head.train()
for ep in range(1, mlm_epochs + 1):
    tot, cnt = 0.0, 0
    for batch in mlm_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim_mlm.zero_grad()
        padmask = batch["input_ids"] == pad_id
        h = encoder(batch["input_ids"], padmask)
        logits = lm_head(h)  # (B,L,V)
        loss = loss_mlm_fn(logits.view(-1, vocab_size), batch["labels_mlm"].view(-1))
        loss.backward()
        optim_mlm.step()
        totsz = (batch["labels_mlm"] != -100).sum().item()
        tot += loss.item() * totsz
        cnt += totsz
    print(f"MLM Epoch {ep}: loss={tot/cnt:.4f}")

# ------------------------------------------------------------------
# Fine-tuning --------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(spr_model.parameters(), lr=2e-4)
max_epochs, patience = 25, 5
best_f1, no_improve = -1, 0
best_state = None


def evaluate(loader):
    spr_model.eval()
    tot_loss = 0
    preds = []
    gts = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pm = batch["input_ids"] == pad_id
            logits = spr_model(batch["input_ids"], pm)
            loss = criterion(logits, batch["label"])
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    loss = tot_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return loss, f1, preds, gts


print("=== Fine-tuning ===")
for epoch in range(1, max_epochs + 1):
    spr_model.train()
    running = 0.0
    for batch in train_loader_cls:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        pm = batch["input_ids"] == pad_id
        logits = spr_model(batch["input_ids"], pm)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["label"].size(0)
    train_loss = running / len(train_loader_cls.dataset)

    val_loss, val_f1, _, _ = evaluate(dev_loader_cls)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
    )

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in spr_model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ------------------------------------------------------------------
# Test evaluation ----------------------------------------------------
spr_model.load_state_dict(best_state)
test_loss, test_f1, test_pred, test_gt = evaluate(test_loader_cls)
print(f"Test: loss={test_loss:.4f}  macro_F1={test_f1:.4f}")

ed["predictions"] = test_pred
ed["ground_truth"] = test_gt

# save metrics -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
