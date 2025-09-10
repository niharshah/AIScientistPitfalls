import os, pathlib, random, math, numpy as np, torch
from typing import Dict, List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ----------------------- working dir & device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------- SPR loader --------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):  # helper
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if spr_path.exists():
    spr = load_spr_bench(spr_path)
else:  # tiny synthetic fallback
    print("SPR_BENCH not found, creating synthetic toy data...")

    def synth(n, L=20, nlab=5):
        data = {"id": [], "sequence": [], "label": []}
        alpha = list("ABCDEUVXYZ")
        for i in range(n):
            ln = random.randint(5, L)
            data["id"].append(str(i))
            data["sequence"].append("".join(random.choices(alpha, k=ln)))
            data["label"].append(random.randint(0, nlab - 1))
        return load_dataset("json", data_files={"train": [data]}, split="train")

    spr = DatasetDict({"train": synth(500), "dev": synth(120), "test": synth(120)})

num_labels = len(set(spr["train"]["label"]))
print(
    f'Dataset sizes – train:{len(spr["train"])}, dev:{len(spr["dev"])}, test:{len(spr["test"])} | labels:{num_labels}'
)

# ----------------------- vocab & encoding --------------------------------
PAD_ID = 0


def build_vocab(ds) -> Dict[str, int]:
    chars = set()
    for s in ds["sequence"]:
        chars.update(s)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = PAD_ID
    return vocab


vocab = build_vocab(spr["train"])
vocab_size = len(vocab)
id2char = {i: c for c, i in vocab.items()}
print("Vocab size:", vocab_size)


def encode(seq: str, max_len: int) -> List[int]:
    ids = [vocab.get(ch, PAD_ID) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


MAX_LEN = min(64, max(len(s) for s in spr["train"]["sequence"]))


# ----------------------- torch dataset -----------------------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, max_len):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor(encode(self.seqs[idx], self.max_len), dtype=torch.long)
        attn = (ids != PAD_ID).long()
        lab = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": lab}


def collate(b):
    return {k: torch.stack([d[k] for d in b]) for k in b[0]}


train_ds, dev_ds, test_ds = (
    SPRTorchDataset(spr[s], MAX_LEN) for s in ["train", "dev", "test"]
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ----------------------- model -------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B,L,D)
        return x + self.pe[:, : x.size(1)]


class SymbolicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        layers=2,
        dim_ff=256,
        drop=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # symbolic histogram pathway
        self.hist_mlp = nn.Sequential(
            nn.Linear(vocab_size, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.classifier = nn.Linear(2 * d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        # contextual branch
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B,d)
        # symbolic histogram branch
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).float()
        mask = attention_mask.unsqueeze(-1)
        hist = (one_hot * mask).sum(1)  # (B,V)
        hist = hist / mask.sum(1).clamp(min=1).float()
        hist_repr = self.hist_mlp(hist)
        # fuse & classify
        fused = torch.cat([pooled, hist_repr], dim=-1)
        return self.classifier(fused)


model = SymbolicTransformer(vocab_size, num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ----------------------- experiment bookkeeping --------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ----------------------- training / eval loops ---------------------------
def run(loader, train_mode: bool):
    model.train() if train_mode else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1, _, _ = run(train_loader, True)
    val_loss, val_f1, _, _ = run(dev_loader, False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={tr_f1:.4f} val_F1={val_f1:.4f}"
    )

# ----------------------- final test --------------------------------------
test_loss, test_f1, test_preds, test_gts = run(test_loader, False)
print(f"Test : loss={test_loss:.4f} MacroF1={test_f1:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
experiment_data["SPR_BENCH"]["test_loss"] = test_loss
experiment_data["SPR_BENCH"]["test_macroF1"] = test_f1

# ----------------------- save artefacts ----------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy in", working_dir)
