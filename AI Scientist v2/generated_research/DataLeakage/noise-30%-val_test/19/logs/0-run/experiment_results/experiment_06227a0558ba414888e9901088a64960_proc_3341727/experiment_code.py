# -------- No-PE ablation study (single-file script) ----------
import os, pathlib, random, math, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- dir / device -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- load (real or synthetic) SPR -----------
def _load_csv_dir(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
    )


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return _load_csv_dir(p)

    # ---------- synthetic fallback ----------
    print("SPR_BENCH not found – generating synthetic toy data")

    def synth(n):
        rows = "ABCD"
        d = []
        for i in range(n):
            seq = "".join(random.choices(rows, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            d.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset("json", data_files={"data": d}, field="data", split="train")

    return DatasetDict({"train": synth(4000), "dev": synth(1000), "test": synth(1000)})


spr = get_spr()

# ---------- vocabulary / encoding ----------
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
    if max_items is not None and len(ds) > max_items:
        ids = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, ids)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# ---------- positional encoding ----------
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


# ---------- Transformer (toggle PE) ----------
class CharTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1, use_pe=True
    ):
        super().__init__()
        self.use_pe = use_pe
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x)
        if self.use_pe:
            h = h + self.pe[:, : x.size(1), :].to(x.device)
        h = self.encoder(h, src_key_padding_mask=mask)
        # mean pooling over non-pad tokens
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0.0).sum(1)) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment dict ----------
experiment_data = {
    "with_positional_encoding": {
        "SPR": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    },
    "no_positional_encoding": {
        "SPR": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        }
    },
}


# ---------- run helper ----------
def run_experiment(use_pe: bool, tag: str, dropout=0.1, epochs=6, lr=1e-3):
    crit = nn.BCEWithLogitsLoss()
    model = CharTransformer(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dropout, use_pe=use_pe
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc, best_state, best_epoch = -1, None, 0

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_preds, tr_lbls = [], [], []
        for batch in train_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["label"])
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())
            tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            tr_lbls.extend(batch["label"].cpu().numpy())
        train_mcc = matthews_corrcoef(tr_lbls, tr_preds)

        # ---- val ----
        model.eval()
        val_loss, val_preds, val_lbls = [], [], []
        with torch.no_grad():
            for batch in dev_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"])
                val_loss.append(crit(logits, batch["label"]).item())
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_lbls.extend(batch["label"].cpu().numpy())
        val_mcc = matthews_corrcoef(val_lbls, val_preds)
        print(f"[{tag}] Epoch {ep}: train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}")

        # log
        d = experiment_data[tag]["SPR"]
        d["metrics"]["train_MCC"].append(train_mcc)
        d["metrics"]["val_MCC"].append(val_mcc)
        d["losses"]["train"].append(np.mean(tr_loss))
        d["losses"]["val"].append(np.mean(val_loss))
        d["epochs"].append(ep)

        if val_mcc > best_mcc:
            best_mcc, best_epoch = val_mcc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[{tag}] Best val_MCC={best_mcc:.3f} at epoch {best_epoch}")
    # ---- test best ----
    best_model = CharTransformer(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dropout, use_pe=use_pe
    ).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in test_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = best_model(batch["input_ids"])
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            gts.extend(batch["label"].cpu().numpy())
    test_mcc = matthews_corrcoef(gts, preds)
    test_f1 = f1_score(gts, preds, average="macro")
    print(f"[{tag}] Test MCC={test_mcc:.3f} | Test F1={test_f1:.3f}")

    d["predictions"] = preds
    d["ground_truth"] = gts
    d["test_MCC"] = test_mcc
    d["test_F1"] = test_f1
    return d


# ---------- run both variants ----------
with_pe_data = run_experiment(use_pe=True, tag="with_positional_encoding")
no_pe_data = run_experiment(use_pe=False, tag="no_positional_encoding")

# ---------- plots ----------
for tag in experiment_data:
    d = experiment_data[tag]["SPR"]
    plt.figure(figsize=(6, 4))
    plt.plot(d["losses"]["train"], label="train")
    plt.plot(d["losses"]["val"], label="val")
    plt.title(f"Loss curve ({tag})")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_{tag}.png"))
    plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
