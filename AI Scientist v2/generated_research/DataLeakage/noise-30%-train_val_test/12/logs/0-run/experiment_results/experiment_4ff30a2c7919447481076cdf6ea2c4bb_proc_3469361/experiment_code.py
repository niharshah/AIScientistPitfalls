import os, pathlib, numpy as np, torch
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ----- housekeeping ----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----- load SPR_BENCH ---------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ----- vocab / encoding -------------------------------------------------------
PAD, UNK = "<pad>", "<unk>"
alphabet = sorted({ch for ex in spr["train"] for ch in ex["sequence"]})
itos = [PAD, UNK] + alphabet
stoi = {ch: i for i, ch in enumerate(itos)}
max_len = 64
num_classes = len(set(spr["train"]["label"]))
vocab_size = len(itos)


def encode(seq: str) -> list:
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode(row["sequence"]), dtype=torch.long)
        mask = (ids != stoi[PAD]).long()
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": label}


# ----- models ----------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, symbolic=False):
        super().__init__()
        d_model = 96
        self.symbolic = symbolic
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        out_dim = d_model * 2 if symbolic else d_model
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        # standard pooled representation
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        if self.symbolic:
            # symbolic Δ-relation: mean consecutive diff
            diff = x[:, 1:, :] - x[:, :-1, :]
            rel = (diff * attention_mask[:, 1:].unsqueeze(-1)).sum(1) / (
                attention_mask[:, 1:].sum(1, keepdim=True) + 1e-6
            )
            pooled = torch.cat([pooled, rel], dim=-1)
        return self.fc(pooled)


# ----- training / evaluation helpers ----------------------------------------
def run_loader(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ----- experiment loop -------------------------------------------------------
experiment_data = {
    "baseline": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "symbolic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

batch_size = 128
epochs = 5
criterion = nn.CrossEntropyLoss()


def train_variant(name, symbolic=False):
    print(f"\n=== {name.upper()} ===")
    model = TinyTransformer(symbolic=symbolic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_loader = DataLoader(
        SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_preds, val_gts = run_loader(model, dev_loader, criterion)
        experiment_data[name]["losses"]["train"].append(tr_loss)
        experiment_data[name]["losses"]["val"].append(val_loss)
        experiment_data[name]["metrics"]["train"].append(tr_f1)
        experiment_data[name]["metrics"]["val"].append(val_f1)
        if epoch == epochs:  # save final predictions
            experiment_data[name]["predictions"] = val_preds
            experiment_data[name]["ground_truth"] = val_gts
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}  val_macroF1={val_f1:.4f}")


train_variant("baseline", symbolic=False)
train_variant("symbolic", symbolic=True)

# ----- save ------------------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nAll done. Results saved to working/experiment_data.npy")
