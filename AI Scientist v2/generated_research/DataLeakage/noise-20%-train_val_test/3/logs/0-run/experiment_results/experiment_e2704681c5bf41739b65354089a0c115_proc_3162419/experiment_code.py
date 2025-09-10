import os, pathlib, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- misc / reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- load SPR ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _ld("train.csv")
    d["dev"] = _ld("dev.csv")
    d["test"] = _ld("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
special_tokens = ["<PAD>"]
chars = sorted({ch for s in spr["train"]["sequence"] for ch in s})
itos = special_tokens + chars
stoi = {t: i for i, t in enumerate(itos)}
pad_id, vocab_size = stoi["<PAD>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
max_len = max(len(s) for s in spr["train"]["sequence"])
print(f"vocab={vocab_size}, classes={num_classes}, max_len={max_len}")


# ---------- dataset ----------
class SPRSet(Dataset):
    def __init__(self, ds):
        self.seqs, self.labels = ds["sequence"], ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "label": label}


def collate(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    # counts vector per sample
    cnts = torch.stack([torch.bincount(s, minlength=vocab_size) for s in seqs]).float()
    return {"input_ids": padded, "counts": cnts, "label": labels}


train_loader = DataLoader(
    SPRSet(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRSet(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRSet(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class CountsContextTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, n_cls, pad, max_len, counts_dim):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.pos_embed = nn.Embedding(max_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.count_proj = nn.Linear(counts_dim, d_model)
        self.classifier = nn.Linear(d_model, n_cls)

    def forward(self, ids, counts):
        b, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(b, L)
        x = self.tok_embed(ids) + self.pos_embed(pos)
        mask = ids.eq(pad_id)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled = x.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1).clamp(
            min=1e-6
        ).unsqueeze(-1)
        cnt_feat = self.count_proj(counts)
        return self.classifier(torch.tanh(pooled + cnt_feat))


model = CountsContextTransformer(
    vocab_size, 192, 6, 3, num_classes, pad_id, max_len, vocab_size
).to(device)

# ---------- training utils ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def run_epoch(loader, train):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["counts"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ---------- train w/ early stopping ----------
best_f1, patience, no_improve = -1.0, 5, 0
for epoch in range(1, 31):
    tr_loss, tr_f1, _, _ = run_epoch(train_loader, True)
    val_loss, val_f1, _, _ = run_epoch(dev_loader, False)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
    )
    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(tr_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)
    if val_f1 > best_f1:
        best_f1, no_improve = val_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ---------- test ----------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, False)
print(f"Test: loss={test_loss:.4f}  macro_f1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# ---------- save experiment ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
