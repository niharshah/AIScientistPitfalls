# Rule-Only Head (Fixed Gate = 1) Ablation
import os, time, pathlib, itertools, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------------- #
#                              HOUSE-KEEPING                                 #
# -------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------------------------------- #
#                             HYPER-PARAMETERS                               #
# -------------------------------------------------------------------------- #
BATCH_SIZE = 256
VAL_BATCH = 512
EPOCHS = 10
LR = 3e-3
WEIGHT_DECAY = 1e-4
L1_LAMBDA = 2e-3
MAX_LEN = 128
PAD_IDX = 0


# -------------------------------------------------------------------------- #
#                              LOAD DATASET                                  #
# -------------------------------------------------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ("train", "dev", "test"):
        d[split] = _load(f"{split}.csv")
    return d


DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATASET_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print("Dataset loaded:", {k: len(v) for k, v in spr.items()})

# -------------------------------------------------------------------------- #
#                               VOCAB                                        #
# -------------------------------------------------------------------------- #
chars = set(itertools.chain.from_iterable(spr["train"]["sequence"]))
char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0: PAD
VOCAB = len(char2idx) + 1


def seq_to_bag(seq: str) -> np.ndarray:
    v = np.zeros(VOCAB - 1, dtype=np.float32)
    for ch in seq[:MAX_LEN]:
        v[char2idx[ch] - 1] += 1
    if len(seq):
        v /= len(seq)
    return v


def seq_to_idx(seq: str) -> np.ndarray:
    arr = np.full(MAX_LEN, PAD_IDX, dtype=np.int64)
    for i, ch in enumerate(seq[:MAX_LEN]):
        arr[i] = char2idx[ch]
    return arr


class SPRDataset(Dataset):
    def __init__(self, split):
        self.bags = np.stack([seq_to_bag(s) for s in split["sequence"]])
        self.seqs = np.stack(
            [seq_to_idx(s) for s in split["sequence"]]
        )  # kept for API compatibility
        self.labels = np.array(split["label"], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.bags[idx]),
            torch.from_numpy(self.seqs[idx]),  # unused in ablation model
            torch.tensor(self.labels[idx]),
        )


train_ds, val_ds, test_ds = [SPRDataset(spr[s]) for s in ("train", "dev", "test")]
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH)
test_loader = DataLoader(test_ds, batch_size=VAL_BATCH)

NUM_CLASSES = int(
    max(train_ds.labels.max(), val_ds.labels.max(), test_ds.labels.max()) + 1
)
print("Vocab", VOCAB - 1, "classes", NUM_CLASSES)


# -------------------------------------------------------------------------- #
#                               MODEL                                        #
# -------------------------------------------------------------------------- #
class RuleOnly(nn.Module):
    """
    Rule-Only Head (gate fixed to 1).  No CNN, no gate parameters.
    """

    def __init__(self, vocab: int, n_cls: int):
        super().__init__()
        self.rule_head = nn.Linear(vocab - 1, n_cls)

    def forward(self, bag, seq=None):  # seq ignored
        logits = self.rule_head(bag)
        return logits  # single output


model = RuleOnly(VOCAB, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -------------------------------------------------------------------------- #
#                           METRIC STORAGE                                   #
# -------------------------------------------------------------------------- #
experiment_data = {
    "rule_only": {
        "SPR_BENCH": {
            "metrics": {"train_acc": [], "val_acc": [], "Rule_Fidelity": []},
            "losses": {"train": [], "val": []},
            "timestamps": [],
            "predictions": [],
            "ground_truth": [],
        }
    }
}


# -------------------------------------------------------------------------- #
#                               EVALUATION                                   #
# -------------------------------------------------------------------------- #
def evaluate(loader):
    model.eval()
    tot, corr, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for bag, seq, y in loader:
            bag, y = bag.to(device), y.to(device)
            logits = model(bag)
            loss = criterion(logits, y)
            corr += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            tot += y.size(0)
    acc = corr / tot
    return acc, loss_sum / tot, 1.0  # fidelity is 1 by construction


# -------------------------------------------------------------------------- #
#                               TRAIN LOOP                                   #
# -------------------------------------------------------------------------- #
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss = run_corr = n_seen = 0
    for bag, seq, y in train_loader:
        bag, y = bag.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(bag)
        ce_loss = criterion(logits, y)
        l1_loss = model.rule_head.weight.abs().mean()
        loss = ce_loss + L1_LAMBDA * l1_loss
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * y.size(0)
        run_corr += (logits.argmax(1) == y).sum().item()
        n_seen += y.size(0)

    train_loss, train_acc = run_loss / n_seen, run_corr / n_seen
    val_acc, val_loss, val_rfs = evaluate(val_loader)

    exp = experiment_data["rule_only"]["SPR_BENCH"]
    exp["losses"]["train"].append(train_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train_acc"].append(train_acc)
    exp["metrics"]["val_acc"].append(val_acc)
    exp["metrics"]["Rule_Fidelity"].append(val_rfs)
    exp["timestamps"].append(time.time())

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
    )

# -------------------------------------------------------------------------- #
#                               FINAL TEST                                   #
# -------------------------------------------------------------------------- #
test_acc, test_loss, test_rfs = evaluate(test_loader)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.3f}, RuleFidelity={test_rfs:.3f}")

# store predictions
model.eval()
preds, gts = [], []
with torch.no_grad():
    for bag, seq, y in test_loader:
        bag = bag.to(device)
        logits = model(bag)
        preds.append(logits.argmax(1).cpu())
        gts.append(y)
exp["predictions"] = torch.cat(preds).numpy()
exp["ground_truth"] = torch.cat(gts).numpy()

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", working_dir)
