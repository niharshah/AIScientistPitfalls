import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------- paths / dirs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------- device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------- experiment store
experiment_data = {}


# ------------------------------------------------- load SPR_BENCH -------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

# ------------------------------ vocab / torch dataset ------------------------------
chars = set("".join(spr["train"]["sequence"]))
vocab = {"<pad>": 0, "<unk>": 1}
vocab.update({ch: i + 2 for i, ch in enumerate(sorted(chars))})
pad_id = vocab["<pad>"]
max_len = min(128, max(len(s) for s in spr["train"]["sequence"]))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:max_len]
        ids = [vocab.get(ch, vocab["<unk>"]) for ch in seq] + [pad_id] * (
            max_len - len(seq)
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size=256)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size=256)

num_classes = len(set(spr["train"]["label"]))


# ------------------------------ model ------------------------------
class ContextualTransformer(nn.Module):
    def __init__(
        self, vocab_size, num_classes, d_model=128, nhead=4, nlayers=2, dropout=0.2
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.context_cnn = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )  # depth-wise
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, num_classes)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: (B,L)
        h = self.token_emb(x) + self.pos[:, : x.size(1)]
        h = self.context_cnn(h.transpose(1, 2)).transpose(1, 2)  # (B,L,d)
        h = self.relu(h)
        h = self.transformer(h.transpose(0, 1)).transpose(0, 1)  # (B,L,d)
        h = self.dropout(h.mean(1))
        return self.cls(h)


# ------------------------------ utils ------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).detach().cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    return (
        tot_loss / len(loader.dataset),
        f1_score(trues, preds, average="macro"),
        preds,
        trues,
    )


# ------------------------------ training loop ------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
search = [{"lr": 1e-3}, {"lr": 5e-4}]
patience = 4
num_epochs = 20

for cfg in search:
    tag = f'lr_{cfg["lr"]}'
    experiment_data[tag] = {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    model = ContextualTransformer(len(vocab), num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_f1, best_state, wait = 0.0, None, 0
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()
        print(
            f"[{tag}] Epoch {epoch}: val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}"
        )
        # log
        experiment_data[tag]["epochs"].append(epoch)
        experiment_data[tag]["losses"]["train"].append(tr_loss)
        experiment_data[tag]["losses"]["val"].append(val_loss)
        experiment_data[tag]["metrics"]["train_macro_f1"].append(tr_f1)
        experiment_data[tag]["metrics"]["val_macro_f1"].append(val_f1)
        # early stop
        if val_f1 > best_f1:
            best_f1, val_best = val_f1, epoch
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(
                f"[{tag}] early stopping at epoch {epoch}, best epoch {val_best} f1={best_f1:.4f}"
            )
            break
    # test with best
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_f1, preds, truths = run_epoch(model, test_loader, criterion)
    print(f"[{tag}] Test macro_f1={test_f1:.4f}")
    experiment_data[tag]["predictions"] = preds
    experiment_data[tag]["ground_truth"] = truths
    experiment_data[tag]["test_macro_f1"] = test_f1
    experiment_data[tag]["test_loss"] = test_loss

# ------------------------------ save all ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
