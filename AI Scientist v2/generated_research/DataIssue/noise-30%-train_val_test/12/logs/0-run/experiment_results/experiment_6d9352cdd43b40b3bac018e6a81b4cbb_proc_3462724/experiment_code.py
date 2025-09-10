import os, pathlib, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
from typing import List

# --------------------- Experiment container ---------------------
experiment_data = {"gradient_clip_max_norm": {}}  # will hold every run
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------- Device ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------- Load SPR-BENCH ---------------------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_file: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_file),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


# try local path first, else download pre-packaged version
try:
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    spr = load_spr_bench(DATA_PATH)
except Exception:
    spr = load_dataset("tner/spr")  # small fallback dataset on HF
    spr = spr.rename_columns({"train": "train", "validation": "dev"})

print("Dataset loaded:", {k: len(v) for k, v in spr.items()})

# --------------------- Vocabulary & encoding ---------------------
PAD, UNK = "<pad>", "<unk>"
char_set = set()
[char_set.update(list(ex["sequence"])) for ex in spr["train"]]
itos = [PAD, UNK] + sorted(char_set)
stoi = {c: i for i, c in enumerate(itos)}


def encode(seq: str, max_len: int = 128):
    ids = [stoi.get(ch, stoi[UNK]) for ch in seq[:max_len]]
    return ids + [stoi[PAD]] * (max_len - len(ids))


max_len = 128
num_classes = len(set(spr["train"]["label"]))


# --------------------- Dataset wrapper ---------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dset, max_len=128):
        self.data = hf_dset
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_ids = torch.tensor(encode(row["sequence"], self.max_len))
        attn_mask = (input_ids != stoi[PAD]).long()
        label = torch.tensor(row["label"])
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": label}


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], max_len), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"], max_len), batch_size=batch_size)


# --------------------- Model ---------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=stoi[PAD])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.fc(x)


# --------------------- Train / eval helpers ---------------------
def run_loader(model, loader, criterion, optimizer=None, clip_val=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.argmax(-1).detach().cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# --------------------- Hyper-parameter sweep ---------------------
clip_values = [0.5, 1.0, 2.0, 5.0, None]
epochs = 5

for clip in clip_values:
    tag = "no_clip" if clip is None else f"clip_{clip}"
    print(f"\n=== Running {tag} ===")
    model = TinyTransformer(len(itos), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    run_data = {
        "epochs": [],
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_loader(
            model, train_loader, criterion, optimizer, clip
        )
        val_loss, val_f1, val_preds, val_gts = run_loader(model, dev_loader, criterion)
        run_data["epochs"].append(ep)
        run_data["losses"]["train"].append(tr_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["train_f1"].append(tr_f1)
        run_data["metrics"]["val_f1"].append(val_f1)
        print(
            f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if ep == epochs:
            run_data["predictions"] = val_preds
            run_data["ground_truth"] = val_gts
    experiment_data["gradient_clip_max_norm"][tag] = run_data
    # plot curves for this run
    plt.figure()
    plt.plot(run_data["epochs"], run_data["losses"]["train"], label="train")
    plt.plot(run_data["epochs"], run_data["losses"]["val"], label="val")
    plt.title(f"Loss ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_{tag}.png"))
    plt.close()
    plt.figure()
    plt.plot(run_data["epochs"], run_data["metrics"]["val_f1"])
    plt.title(f"Val Macro-F1 ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.savefig(os.path.join(working_dir, f"f1_{tag}.png"))
    plt.close()
    del model, optimizer
    torch.cuda.empty_cache()

# --------------------- Save experiment data ---------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(
    "\nFinished all runs. Saved experiment data to",
    os.path.join(working_dir, "experiment_data.npy"),
)
