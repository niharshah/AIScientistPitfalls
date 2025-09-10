import os, pathlib, random, math, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, f1_score
import matplotlib.pyplot as plt

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- SPR loading ----------
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
    # fallback synthetic
    print("SPR_BENCH not found, generating synthetic toy data")

    def synth(n):
        rows = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(rows, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
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
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for PAD
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
        idx = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", max_items=2000)
test_loader = lambda: make_loader("test")


# ---------- positional encoding helpers ----------
def sinusoidal_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, seq, d)


# ---------- model ----------
class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pe_type="sinusoidal",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe_type = pe_type
        if pe_type == "sinusoidal":
            self.register_buffer(
                "pe", sinusoidal_encoding(max_len, d_model, torch.device("cpu"))
            )
        else:  # learned positional embedding
            self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mask = x.eq(0)
        tok = self.emb(x)
        if self.pe_type == "sinusoidal":
            pos = self.pe[:, : x.size(1)].to(x.device)
        else:
            idx = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            pos = self.pos_emb(idx)
        h = tok + pos
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h.masked_fill(mask.unsqueeze(2), 0.0).sum(1)) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


# ---------- experiment setup ----------
experiment_data = {
    "positional_encoding": {
        "sinusoidal": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        },
        "learned": {
            "metrics": {"train_MCC": [], "val_MCC": []},
            "losses": {"train": [], "val": []},
            "epochs": [],
            "predictions": [],
            "ground_truth": [],
        },
    }
}

criterion = nn.BCEWithLogitsLoss()
dropouts = [0.1, 0.3]
epochs = 6
pe_variants = ["sinusoidal", "learned"]

for variant in pe_variants:
    print(f"\n### Positional Encoding: {variant.upper()} ###")
    best_dev_mcc, best_state, best_dp = -1, None, None
    for dp in dropouts:
        print(f"\n--- Dropout {dp} ---")
        model = CharTransformer(
            vocab_size, d_model=128, nhead=4, num_layers=2, dropout=dp, pe_type=variant
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(1, epochs + 1):
            # ----- train -----
            model.train()
            tr_losses, tr_preds, tr_lbls = [], [], []
            for batch in train_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad()
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["label"])
                loss.backward()
                optim.step()
                tr_losses.append(loss.item())
                tr_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                tr_lbls.extend(batch["label"].cpu().numpy())
            train_mcc = matthews_corrcoef(tr_lbls, tr_preds)

            # ----- validation -----
            model.eval()
            val_losses, val_preds, val_lbls = [], [], []
            with torch.no_grad():
                for batch in dev_loader():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(batch["input_ids"])
                    val_losses.append(criterion(logits, batch["label"]).item())
                    val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                    val_lbls.extend(batch["label"].cpu().numpy())
            val_mcc = matthews_corrcoef(val_lbls, val_preds)
            print(
                f"Epoch {ep}: val_loss={np.mean(val_losses):.4f} | "
                f"train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
            )

            # record
            exp = experiment_data["positional_encoding"][variant]
            exp["metrics"]["train_MCC"].append(train_mcc)
            exp["metrics"]["val_MCC"].append(val_mcc)
            exp["losses"]["train"].append(np.mean(tr_losses))
            exp["losses"]["val"].append(np.mean(val_losses))
            exp["epochs"].append((dp, ep))

            if val_mcc > best_dev_mcc:
                best_dev_mcc = val_mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_dp = dp

    # ---------- test best checkpoint ----------
    print(
        f"\nBest dev MCC for {variant}: {best_dev_mcc:.3f} (dropout={best_dp}). "
        "Evaluating on test set..."
    )
    best_model = CharTransformer(
        vocab_size, d_model=128, nhead=4, num_layers=2, dropout=best_dp, pe_type=variant
    ).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    test_preds, test_lbls = [], []
    with torch.no_grad():
        for batch in test_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = best_model(batch["input_ids"])
            test_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            test_lbls.extend(batch["label"].cpu().numpy())
    test_mcc = matthews_corrcoef(test_lbls, test_preds)
    test_f1 = f1_score(test_lbls, test_preds, average="macro")
    print(f"Test MCC={test_mcc:.3f} | Test Macro-F1={test_f1:.3f}")

    exp = experiment_data["positional_encoding"][variant]
    exp["predictions"] = test_preds
    exp["ground_truth"] = test_lbls
    exp["test_MCC"] = test_mcc
    exp["test_F1"] = test_f1

    # ---------- plot ----------
    plt.figure(figsize=(6, 4))
    plt.plot(exp["losses"]["train"], label="train")
    plt.plot(exp["losses"]["val"], label="val")
    plt.xlabel("update (epochs aggregated)")
    plt.ylabel("BCE loss")
    plt.title(f"Loss curve ({variant})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{variant}.png"))
    plt.close()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
