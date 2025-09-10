import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, matthews_corrcoef
import matplotlib.pyplot as plt

# -------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------- SPR loading -----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _ld("train.csv"), _ld("dev.csv"), _ld("test.csv")
    return d


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)

    # synthetic fallback if real data absent
    print("SPR_BENCH not found, generating synthetic toy data")

    def synth(n):
        rows, shapes = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            lbl = int(
                seq.count("A") % 2 == 0 and seq[-1] in "BC"
            )  # slightly harder rule
            data.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = synth(4000), synth(1000), synth(1000)
    return d


spr = get_spr()

# ------------- vocab & encoding --------------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = {i: ch for ch, i in enumerate(["<PAD>"] + vocab)}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.float),
        }


def make_loader(name, bs=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        ids = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, ids)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


# Sub-sample to keep runtime reasonable on large data
train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# ---------- sinusoidal positional encoding ----------
def positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, seq, d)


# ---------- Transformer model -------------
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pe[:, : x.size(1), :].to(x.device)
        h = self.encoder(h, src_key_padding_mask=mask)
        # mean over non-pad tokens
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0.0).sum(1)) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment setup ----------
experiment_data = {
    "transformer": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

dropouts = [0.1, 0.3]
best_dev_mcc = -1
best_state = None
criterion = nn.BCEWithLogitsLoss()
epochs = 6

for dp in dropouts:
    print(f"\n=== Dropout {dp} ===")
    model = CharTransformer(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dp
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_losses, tr_preds, tr_labels = [], [], []
        for batch in train_loader():
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_labels.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_labels, tr_preds)

        # validation
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                logits = model(batch["input_ids"])
                val_losses.append(criterion(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_labels, val_preds)
        print(
            f"Epoch {epoch}: validation_loss = {np.mean(val_losses):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
        )

        # store
        experiment_data["transformer"]["metrics"]["train_MCC"].append(train_mcc)
        experiment_data["transformer"]["metrics"]["val_MCC"].append(val_mcc)
        experiment_data["transformer"]["losses"]["train"].append(np.mean(tr_losses))
        experiment_data["transformer"]["losses"]["val"].append(np.mean(val_losses))
        experiment_data["transformer"]["epochs"].append((dp, epoch))

        # keep best
        if val_mcc > best_dev_mcc:
            best_dev_mcc = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_dp = dp

# ---------- test with best model ----------
print(
    f"\nBest dev MCC={best_dev_mcc:.3f} (dropout={best_dp}). Evaluating on test set..."
)
best_model = CharTransformer(
    vocab_size, d_model=128, nhead=4, num_layers=2, dropout=best_dp
).to(device)
best_model.load_state_dict(best_state)
best_model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader():
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = best_model(batch["input_ids"])
        test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
        test_labels.extend(batch["label"].cpu().numpy())
test_mcc = matthews_corrcoef(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average="macro")
print(f"Test MCC={test_mcc:.3f} | Test Macro-F1={test_f1:.3f}")

experiment_data["transformer"]["predictions"] = test_preds
experiment_data["transformer"]["ground_truth"] = test_labels
experiment_data["transformer"]["test_MCC"] = test_mcc
experiment_data["transformer"]["test_F1"] = test_f1

# --------- plots ----------
plt.figure(figsize=(6, 4))
plt.plot(experiment_data["transformer"]["losses"]["train"], label="train")
plt.plot(experiment_data["transformer"]["losses"]["val"], label="val")
plt.xlabel("update (epochs aggregated)")
plt.ylabel("BCE loss")
plt.legend()
plt.title("Loss curve Transformer")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "loss_curve_transformer.png"))
plt.close()

# -------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
