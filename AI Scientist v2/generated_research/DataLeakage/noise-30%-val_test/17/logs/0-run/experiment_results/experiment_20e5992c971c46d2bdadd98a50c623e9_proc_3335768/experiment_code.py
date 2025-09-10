import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------#
# Reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ------------------------------------------------------------------#
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)


# ------------------- Vocabulary & Encoding ------------------------#
def build_vocab(dsets):
    chars = set()
    for split in dsets.values():
        for s in split["sequence"]:
            chars.update(s)
    # 0 = PAD, 1 = CLS
    return {ch: i + 2 for i, ch in enumerate(sorted(chars))}


vocab = build_vocab(spr)
CLS_ID = 1
PAD_ID = 0
vocab_size = len(vocab) + 2  # +PAD +CLS
max_len = (
    max(max(len(s) for s in split["sequence"]) for split in spr.values()) + 1
)  # +CLS


def encode_sequence(seq: str):
    return [CLS_ID] + [vocab[ch] for ch in seq][: max_len - 1]


def pad(seq_ids: list[int]):
    L = max_len
    seq_ids = seq_ids[:L]
    return seq_ids + [PAD_ID] * (L - len(seq_ids))


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = pad(encode_sequence(self.seqs[idx]))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


batch_size = 128
train_loader = DataLoader(SPRTorchDataset(spr["train"]), batch_size, shuffle=True)
dev_loader = DataLoader(SPRTorchDataset(spr["dev"]), batch_size)
test_loader = DataLoader(SPRTorchDataset(spr["test"]), batch_size)


# ----------------------- Model ------------------------------------#
class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids):
        mask = input_ids == PAD_ID
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=mask)
        cls_repr = x[:, 0, :]  # representation of CLS token
        logits = self.cls_head(cls_repr).squeeze(1)
        return logits


# ----------------------- Utils ------------------------------------#
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best, self.counter, self.stop = None, 0, False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric - self.best) if self.mode == "max" else (self.best - metric)
        if improve > self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "configs": [],
    }
}


def evaluate(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            total_loss += criterion(logits, batch["labels"]).item() * batch[
                "labels"
            ].size(0)
            preds.append((logits.sigmoid() > 0.5).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    loss = total_loss / len(loader.dataset)
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return loss, mcc, f1, preds, labels


def train_run(epochs=12, lr=2e-4, patience=4, weight_decay=1e-4):
    model = TinyTransformerClassifier(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=patience, mode="max")
    best_state, best_f1 = None, -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["labels"].size(0)
        train_loss = running_loss / len(train_loader.dataset)
        _, train_mcc, train_f1, _, _ = evaluate(model, train_loader)
        val_loss, val_mcc, val_f1, _, _ = evaluate(model, dev_loader)
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f}"
        )

        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_f1)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
        if es(val_f1):
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    test_loss, test_mcc, test_f1, preds, labels = evaluate(model, test_loader)
    print(f"Test macro_F1 = {test_f1:.4f} | Test MCC = {test_mcc:.4f}")
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(labels)
    experiment_data["SPR_BENCH"]["configs"].append(
        {"epochs": epochs, "lr": lr, "patience": patience}
    )


# ----------------------- Hyper-parameter sweep --------------------#
for lr in [2e-4, 5e-4]:
    print(f"\n=== Training Transformer | lr={lr} ===")
    train_run(epochs=12, lr=lr, patience=3)

# ----------------------- Save -------------------------------------#
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
