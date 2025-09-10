import os, pathlib, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ────────────────────────── housekeeping ──────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# container that will be saved to disk
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "hyperparams": {},
    }
}


# ────────────────────────── dataset utils ──────────────────────────
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.pad_id, self.max_len = vocab, vocab["<pad>"], max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in seq[: self.max_len]]
        ids += [self.pad_id] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ────────────────────────── model ──────────────────────────
class SPRModel(nn.Module):
    def __init__(
        self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, max_len=128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, : x.size(1)]
        x = self.encoder(x.transpose(0, 1)).transpose(0, 1)
        return self.cls(x.mean(1))


# ────────────────────────── train / eval helpers ──────────────────────────
def move_batch(batch):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    tl, preds, gts = 0.0, [], []
    for batch in loader:
        batch = move_batch(batch)
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        tl += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    return tl / len(loader.dataset), f1_score(gts, preds, average="macro")


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tl, preds, gts = 0.0, [], []
    for batch in loader:
        batch = move_batch(batch)
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        tl += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    return tl / len(loader.dataset), f1_score(gts, preds, average="macro"), preds, gts


# ────────────────────────── experiment runner ──────────────────────────
def run_experiment():
    # path to SPR_BENCH (adjust if necessary)
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset path {DATA_PATH} not found.")
    spr = load_spr_bench(DATA_PATH)

    # build vocabulary
    chars = set("".join(spr["train"]["sequence"]))
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        **{ch: i + 2 for i, ch in enumerate(sorted(chars))},
    }
    max_len = min(128, max(len(s) for s in spr["train"]["sequence"]))

    # torch datasets / loaders (batch size tuned later)
    train_ds = SPRTorchDataset(spr["train"], vocab, max_len)
    val_ds = SPRTorchDataset(spr["dev"], vocab, max_len)
    test_ds = SPRTorchDataset(spr["test"], vocab, max_len)

    num_classes = len(set(spr["train"]["label"]))
    criterion = nn.CrossEntropyLoss()

    hyper_grid = [
        {"d_model": 128, "lr": 1e-3, "batch": 128},
        {"d_model": 256, "lr": 5e-4, "batch": 128},
    ]

    for hp in hyper_grid:
        print(f"\n──── Hyper-params: {hp} ────")
        experiment_data["SPR_BENCH"]["hyperparams"][str(hp)] = {}

        # loaders with chosen batch size
        train_loader = DataLoader(train_ds, batch_size=hp["batch"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hp["batch"] * 2)
        test_loader = DataLoader(test_ds, batch_size=hp["batch"] * 2)

        model = SPRModel(
            len(vocab), num_classes, d_model=hp["d_model"], max_len=max_len
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=False
        )

        best_val, patience, max_patience = 0.0, 0, 3
        for epoch in range(1, 11):  # up to 10 epochs
            tr_loss, tr_f1 = train_epoch(model, train_loader, criterion, optimizer)
            vl_loss, vl_f1, _, _ = eval_epoch(model, val_loader, criterion)
            scheduler.step(vl_f1)

            # record
            experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
            experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
            experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_f1)
            experiment_data["SPR_BENCH"]["metrics"]["val"].append(vl_f1)
            experiment_data["SPR_BENCH"]["epochs"].append(
                {"epoch": epoch, "time": time.time(), "hp": hp}
            )

            print(
                f"Epoch {epoch:02d} | val_loss={vl_loss:.4f} | val_macro_f1={vl_f1:.4f}"
            )
            if vl_f1 > best_val:
                best_val, patience = vl_f1, 0
                torch.save(
                    model.state_dict(),
                    os.path.join(working_dir, f'best_model_{hp["d_model"]}.pt'),
                )
            else:
                patience += 1
            if patience >= max_patience:
                print("Early stopping triggered.")
                break

        # load the best weights before final test
        model.load_state_dict(
            torch.load(os.path.join(working_dir, f'best_model_{hp["d_model"]}.pt'))
        )
        te_loss, te_f1, preds, gts = eval_epoch(model, test_loader, criterion)
        print(f"Test Macro-F1 (d_model={hp['d_model']}) = {te_f1:.4f}")

        # store test data
        experiment_data["SPR_BENCH"]["predictions"].append(preds)
        experiment_data["SPR_BENCH"]["ground_truth"].append(gts)
        experiment_data["SPR_BENCH"]["hyperparams"][str(hp)]["test_macro_f1"] = te_f1

    # persist everything
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print("All experiment data saved.")


# ────────────────────────── run ──────────────────────────
run_experiment()
