import os, pathlib, torch, numpy as np, time, math, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------#
# working directory & device                                                   #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------#
# experiment data container                                                    #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# -----------------------------------------------------------------------------#
# locate SPR_BENCH                                                             #
def _find_spr_bench() -> pathlib.Path:
    for p in [
        pathlib.Path(os.getenv("SPR_DATA", "")),
        pathlib.Path("./SPR_BENCH").resolve(),
        pathlib.Path("../SPR_BENCH").resolve(),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH").resolve(),
    ]:
        if p and p.exists():
            needed = {"train.csv", "dev.csv", "test.csv"}
            if needed.issubset({f.name for f in p.iterdir()}):
                print(f"Found SPR_BENCH at {p}")
                return p
    raise FileNotFoundError("SPR_BENCH not found; set SPR_DATA env var")


# -----------------------------------------------------------------------------#
# dataset                                                                      #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):  # hugface CSV loader
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ("train", "dev", "test")})


def build_vocab(train_split):
    chars = {ch for ex in train_split for ch in ex["sequence"].replace(" ", "")}
    vocab = {"<pad>": 0}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab


class SPRCharDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.data = hf_ds
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]

    def __len__(self):
        return len(self.data)

    def _encode(self, seq):
        seq = seq.replace(" ", "")
        return torch.tensor([self.vocab[ch] for ch in seq], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_tensor = self._encode(row["sequence"])
        lbl = int(row["label"])
        # auxiliary targets
        parity = len(seq_tensor) % 2  # 0 even, 1 odd
        uniq_bucket = min(len(set(seq_tensor.tolist())), 10) - 1  # 0-9
        return {
            "input_ids": seq_tensor,
            "labels": torch.tensor(lbl, dtype=torch.long),
            "parity": torch.tensor(parity, dtype=torch.long),
            "uniq": torch.tensor(uniq_bucket, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    keys = ["input_ids", "labels", "parity", "uniq"]
    seqs = [b["input_ids"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    attention = (padded != pad_id).long()
    collated = {
        "input_ids": padded,
        "attention_mask": attention,
    }
    for k in keys[1:]:
        collated[k] = torch.stack([b[k] for b in batch])
    return collated


# -----------------------------------------------------------------------------#
# model                                                                        #
class MultiTaskTransformer(nn.Module):
    def __init__(
        self, vocab, d_model, nhead, nlayers, num_labels, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.pool = lambda x, m: (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        self.main_head = nn.Linear(d_model, num_labels)
        self.parity_head = nn.Linear(d_model, 2)
        self.uniq_head = nn.Linear(d_model, 10)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.embed(input_ids) + self.pos_embed[:seq_len].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        pooled = self.pool(x, attention_mask)
        return {
            "main": self.main_head(pooled),
            "parity": self.parity_head(pooled),
            "uniq": self.uniq_head(pooled),
        }


# -----------------------------------------------------------------------------#
# training / evaluation                                                        #
def run_epoch(model, loader, crit_main, crit_aux, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    epoch_loss, y_true, y_pred = 0.0, [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            out = model(batch["input_ids"], batch["attention_mask"])
            loss_main = crit_main(out["main"], batch["labels"])
            loss_parity = crit_aux(out["parity"], batch["parity"])
            loss_uniq = crit_aux(out["uniq"], batch["uniq"])
            loss = loss_main + 0.2 * (loss_parity + loss_uniq)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_loss += loss.item() * batch["labels"].size(0)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(out["main"].argmax(1).cpu().tolist())
    avg_loss = epoch_loss / len(loader.dataset)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, macro_f1, y_true, y_pred


# -----------------------------------------------------------------------------#
# pipeline                                                                     #
data_dir = _find_spr_bench()
spr = load_spr_bench(data_dir)
vocab = build_vocab(spr["train"])
max_len = max(len(ex["sequence"].replace(" ", "")) for ex in spr["train"])
num_labels = len(set(int(ex["label"]) for ex in spr["train"]))

train_ds = SPRCharDataset(spr["train"], vocab)
dev_ds = SPRCharDataset(spr["dev"], vocab)
test_ds = SPRCharDataset(spr["test"], vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, vocab["<pad>"]),
)

model = MultiTaskTransformer(
    vocab,
    d_model=96,
    nhead=4,
    nlayers=3,
    num_labels=num_labels,
    max_len=max_len,
    dropout=0.1,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit_main = nn.CrossEntropyLoss()
crit_aux = nn.CrossEntropyLoss()

best_f1, best_state = 0.0, None
epochs = 5
for epoch in range(1, epochs + 1):
    t_loss, t_f1, _, _ = run_epoch(model, train_loader, crit_main, crit_aux, optimizer)
    v_loss, v_f1, _, _ = run_epoch(model, dev_loader, crit_main, crit_aux)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(t_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(v_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(t_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(v_f1)
    print(f"Epoch {epoch}: validation_loss = {v_loss:.4f}, Macro-F1 = {v_f1:.4f}")
    if v_f1 > best_f1:
        best_f1, best_state = v_f1, {
            k: v.clone().detach().cpu() for k, v in model.state_dict().items()
        }

# restore best
model.load_state_dict(best_state)

# final test
test_loss, test_f1, gts, preds = run_epoch(model, test_loader, crit_main, crit_aux)
print(f"Test Macro-F1: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved.")
