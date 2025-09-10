# No-FeedForward-Ablation : single-file run script
import os, pathlib, numpy as np, torch, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
from typing import List, Dict

# ---------------- SETTINGS ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------- DATA ---------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if not root.exists():  # fallback: download public copy
        return load_dataset(
            "gsm8k", split="train[:1%]"
        )  # dummy tiny set if path missing

    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    [d.__setitem__(sp, _load(f"{sp}.csv")) for sp in ["train", "dev", "test"]]
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# --------------- VOCAB --------------------
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {c: i for i, c in enumerate(itos)}
max_len, num_classes = 128, len(set(spr["train"]["label"]))


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


class SPRTorchDataset(Dataset):
    def __init__(self, hf, max_len=128):
        self.data, self.max_len = hf, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        inp = torch.tensor(encode(row["sequence"], self.max_len), dtype=torch.long)
        mask = (inp != stoi[PAD]).long()
        return {
            "input_ids": inp,
            "attention_mask": mask,
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


# ------------- NO-FFN ENCODER -------------
class NoFFNEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, drop: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=drop, batch_first=True
        )
        self.dropout1 = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        attn, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn))
        return src


# ------------- MODEL ----------------------
class TinyTransformerNoFFN(nn.Module):
    def __init__(self, vocab: int, classes: int, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        layers = [NoFFNEncoderLayer(d_model, n_heads) for _ in range(n_layers)]
        self.encoder = nn.ModuleList(layers)
        self.fc = nn.Linear(d_model, classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        pad_mask = ~attention_mask.bool()
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=pad_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(x)


# ------------- TRAIN / EVAL ---------------
def run_loader(model, loader, criterion, opt=None):
    model.train() if opt else model.eval()
    tot, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["input_ids"], b["attention_mask"])
            loss = criterion(out, b["labels"])
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * b["labels"].size(0)
            preds += out.argmax(-1).cpu().tolist()
            gts += b["labels"].cpu().tolist()
    return tot / len(loader.dataset), f1_score(gts, preds, average="macro"), preds, gts


# ------------- EXPERIMENT -----------------
experiment_data = {"no_ffn": {"SPR-BENCH": {"batch_size": {}}}}
batch_sizes = [32, 64, 128, 256]
epochs = 5
criterion = nn.CrossEntropyLoss()

for bs in batch_sizes:
    print(f"\n== BS {bs} ==")
    tr_loader = DataLoader(
        SPRTorchDataset(spr["train"], max_len), batch_size=bs, shuffle=True
    )
    dv_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=bs)
    model = TinyTransformerNoFFN(len(itos), num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = {
        "epochs": [],
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(model, tr_loader, criterion, opt)
        dv_loss, dv_f1, preds, gts = run_loader(model, dv_loader, criterion)
        stats["epochs"].append(ep)
        stats["losses"]["train"].append(tr_loss)
        stats["losses"]["val"].append(dv_loss)
        stats["metrics"]["train"].append(tr_f1)
        stats["metrics"]["val"].append(dv_f1)
        if ep == epochs:
            stats["predictions"], stats["ground_truth"] = preds, gts
        print(f"Ep{ep} TL{tr_loss:.3f} VL{dv_loss:.3f} VF1{dv_f1:.3f}")
    # plots
    plt.figure()
    plt.plot(stats["epochs"], stats["losses"]["train"], label="train")
    plt.plot(stats["epochs"], stats["losses"]["val"], label="val")
    plt.legend()
    plt.title(f"Loss bs{bs}")
    plt.savefig(os.path.join(working_dir, f"loss_bs{bs}.png"))
    plt.close()
    plt.figure()
    plt.plot(stats["epochs"], stats["metrics"]["val"], label="val_f1")
    plt.savefig(os.path.join(working_dir, f"f1_bs{bs}.png"))
    plt.close()
    experiment_data["no_ffn"]["SPR-BENCH"]["batch_size"][bs] = stats

# ------------- SAVE -----------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to working/experiment_data.npy")
