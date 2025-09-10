import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset
from sklearn.metrics import f1_score
from typing import List

# ----------------------------------------------------------------- EXPERIMENT LOG
experiment_data = {"d_model": {"SPR_BENCH": {}}}  # hyper-parameter we tune  # dataset
# ----------------------------------------------------------------- DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ----------------------------------------------------------------- LOAD DATA
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ----------------------------------------------------------------- VOCAB & ENCODING
PAD, UNK = "<pad>", "<unk>"
char_set = set(ch for ex in spr["train"] for ch in ex["sequence"])
itos = [PAD, UNK] + sorted(list(char_set))
stoi = {ch: i for i, ch in enumerate(itos)}


def encode(seq: str, max_len: int = 128) -> List[int]:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


max_len = 128
num_classes = len(set(spr["train"]["label"]))


# ----------------------------------------------------------------- DATASET WRAPPER
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data, self.max_len = hf_dataset, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode(row["sequence"], self.max_len), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": label}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=batch_size)


# ----------------------------------------------------------------- MODEL DEF
class TinyTransformer(nn.Module):
    def __init__(self, vocab, classes, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=stoi[PAD])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(x)


# ----------------------------------------------------------------- TRAIN / EVAL HELPERS
criterion = nn.CrossEntropyLoss()


def run_loader(model, loader, train=False, optimizer=None):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).detach().cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ----------------------------------------------------------------- HYPERPARAM SEARCH
hidden_sizes = [64, 128, 192, 256, 384]
epochs = 5

for d in hidden_sizes:
    print(f"\n=== Training with d_model={d} ===")
    model = TinyTransformer(len(itos), num_classes, d_model=d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # allocate dict for this setting
    experiment_data["d_model"]["SPR_BENCH"][str(d)] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    log = experiment_data["d_model"]["SPR_BENCH"][str(d)]

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(model, train_loader, True, optimizer)
        val_loss, val_f1, vp, vg = run_loader(model, dev_loader, False)

        log["epochs"].append(epoch)
        log["losses"]["train"].append(tr_loss)
        log["losses"]["val"].append(val_loss)
        log["metrics"]["train_f1"].append(tr_f1)
        log["metrics"]["val_f1"].append(val_f1)
        if epoch == epochs:
            log["predictions"], log["ground_truth"] = vp, vg

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

# ----------------------------------------------------------------- SAVE RESULTS
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# optional quick summary
best = max(
    (max(v["metrics"]["val_f1"]), k)
    for k, v in experiment_data["d_model"]["SPR_BENCH"].items()
)
print(f"\nBest validation Macro-F1 {best[0]:.4f} achieved with d_model={best[1]}")
