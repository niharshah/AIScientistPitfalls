import os, pathlib, random, string, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef as mcc

# ─── directory for outputs ───────────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ─── device handling ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── load SPR_BENCH (falls back to synthetic) ───────────────────────────────────
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = {}
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    dsets = load_spr_bench(data_path) if data_path.exists() else None
except FileNotFoundError:
    dsets = None
if dsets is None:
    from datasets import Dataset, DatasetDict

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(6, 16)
            s = "".join(random.choices(list(string.ascii_lowercase) + "#@&", k=L))
            labels.append(int(s.count("#") % 2 == 0))
            seqs.append(s)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict({"train": synth(2000), "dev": synth(400), "test": synth(400)})
print({k: len(v) for k, v in dsets.items()})

# ─── vocab & encoding ───────────────────────────────────────────────────────────
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(c, 1) for c in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ─── dataloader ────────────────────────────────────────────────────────────────
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), 0, dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq)
    return {"input_ids": padded, "labels": labels}


train_loader = DataLoader(
    dsets["train"], batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dsets["dev"], batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    dsets["test"], batch_size=256, shuffle=False, collate_fn=collate
)


# ─── model ─────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerSPR(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, x):
        mask = x == 0
        y = self.pos(self.emb(x))
        h = self.enc(y, src_key_padding_mask=mask)
        h = h.max(dim=1).values  # max-pool over time
        return self.cls(h)


# ─── experiment tracking dict ──────────────────────────────────────────────────
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ─── grid over encoder layers ──────────────────────────────────────────────────
layer_options = [1, 2]
best_dev_mcc, best_state = -1.0, None

for num_layers in layer_options:
    print(f"\n=== training Transformer with {num_layers} encoder layer(s) ===")
    model = TransformerSPR(vocab_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        # ---- train ----
        model.train()
        tr_losses = []
        tr_preds = []
        tr_gts = []
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend(logits.argmax(1).cpu().numpy())
            tr_gts.extend(batch["labels"].cpu().numpy())
        tr_f1 = f1_score(tr_gts, tr_preds, average="macro")
        tr_mcc = mcc(tr_gts, tr_preds)

        # ---- validate ----
        model.eval()
        dv_losses = []
        dv_preds = []
        dv_gts = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = loss_fn(logits, batch["labels"])
                dv_losses.append(loss.item())
                dv_preds.extend(logits.argmax(1).cpu().numpy())
                dv_gts.extend(batch["labels"].cpu().numpy())
        dv_f1 = f1_score(dv_gts, dv_preds, average="macro")
        dv_mcc = mcc(dv_gts, dv_preds)
        print(
            f"Epoch {epoch}: validation_loss = {np.mean(dv_losses):.4f} | "
            f"Val F1={dv_f1:.4f} | Val MCC={dv_mcc:.4f}"
        )

        # log
        ed = experiment_data["SPR_BENCH"]
        ed["metrics"]["train"].append({"f1": tr_f1, "mcc": tr_mcc})
        ed["metrics"]["val"].append({"f1": dv_f1, "mcc": dv_mcc})
        ed["losses"]["train"].append(np.mean(tr_losses))
        ed["losses"]["val"].append(np.mean(dv_losses))
        ed["epochs"].append({"layer_cfg": num_layers, "epoch": epoch})

        # keep best
        if dv_mcc > best_dev_mcc:
            best_dev_mcc = dv_mcc
            best_state = {"model": model.state_dict(), "cfg": num_layers}

# ─── test evaluation with best model ───────────────────────────────────────────
print(
    f'\nBest dev MCC={best_dev_mcc:.4f} with {best_state["cfg"]} layer(s). Evaluating on test…'
)
best_model = TransformerSPR(vocab_size, num_layers=best_state["cfg"]).to(device)
best_model.load_state_dict(best_state["model"])
best_model.eval()
ts_preds = []
ts_gts = []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = best_model(batch["input_ids"])
        ts_preds.extend(logits.argmax(1).cpu().numpy())
        ts_gts.extend(batch["labels"].cpu().numpy())
test_f1 = f1_score(ts_gts, ts_preds, average="macro")
test_mcc = mcc(ts_gts, ts_preds)
print(f"Test F1={test_f1:.4f} | Test MCC={test_mcc:.4f}")

# store final preds/labels
ed = experiment_data["SPR_BENCH"]
ed["predictions"] = ts_preds
ed["ground_truth"] = ts_gts
ed["test_metrics"] = {"f1": test_f1, "mcc": test_mcc}

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("All metrics saved to working/experiment_data.npy")
