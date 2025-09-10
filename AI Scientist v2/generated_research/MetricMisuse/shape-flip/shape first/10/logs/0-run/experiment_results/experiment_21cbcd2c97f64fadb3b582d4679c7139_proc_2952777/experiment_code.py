import os, pathlib, random, math, copy, warnings, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ----------------------- mandatory working dir ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- device handling ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------- reproducibility ------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")


# ----------------------- metrics --------------------------------------------
def _count_shape(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def _count_color(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def build_train_token_set(train_seqs):
    tokens = set()
    for s in train_seqs:
        tokens.update(s.split())
    return tokens


def novelty_weighted_accuracy(seqs, y_true, y_pred, train_token_set):
    weights, correct = [], []
    for s, t, p in zip(seqs, y_true, y_pred):
        toks = set(s.split())
        novelty = len([tok for tok in toks if tok not in train_token_set])
        novelty = max(novelty, 1)  # avoid zero weight
        weights.append(novelty)
        correct.append(novelty if t == p else 0)
    return sum(correct) / max(sum(weights), 1)


# ----------------------- dataset helpers ------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        print("Trying to load SPR_BENCH ...")
        return load_spr_bench(DATA_PATH)
    except Exception as e:
        print("Dataset not found, generating synthetic data.", e)
        shapes, colors = "ABCD", "abcd"

        def make(n):
            data = [
                {
                    "id": i,
                    "sequence": " ".join(
                        random.choice(shapes) + random.choice(colors)
                        for _ in range(random.randint(3, 10))
                    ),
                    "label": random.choice(["yes", "no"]),
                }
                for i in range(n)
            ]
            return load_dataset("json", data_files={"train": data}, split="train")

        return DatasetDict(
            {"train": make(20000), "dev": make(5000), "test": make(10000)}
        )


spr = get_dataset()

# ----------------------- label maps -----------------------------------------
label2id = {l: i for i, l in enumerate(sorted({ex["label"] for ex in spr["train"]}))}
id2label = {v: k for k, v in label2id.items()}


# ----------------------- torch dataset --------------------------------------
class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.lbl = [label2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        n_shape = _count_shape(seq)
        n_color = _count_color(seq)
        length = len(seq.split())
        # simple normalisation
        feats = torch.tensor(
            [n_shape / length, n_color / length, (n_shape * n_color) / (length**2)],
            dtype=torch.float,
        )
        return {
            "sym": feats,
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "raw_seq": seq,
        }


def collate(batch):
    return {
        "sym": torch.stack([b["sym"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(spr["test"]), batch_size, shuffle=False, collate_fn=collate
)


# ----------------------- model ----------------------------------------------
class SymbolicOnlyClassifier(nn.Module):
    def __init__(self, sym_dim=3, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sym_dim, 16), nn.ReLU(), nn.Linear(16, n_cls)
        )

    def forward(self, x):
        return self.net(x)


model = SymbolicOnlyClassifier(sym_dim=3, n_cls=len(label2id)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------- experiment storage ---------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ----------------------- helpers --------------------------------------------
train_token_set = build_train_token_set(spr["train"]["sequence"])


def evaluate(loader):
    model.eval()
    tot_loss = n_items = 0
    y_true = []
    y_pred = []
    raw_seqs = []
    with torch.no_grad():
        for batch in loader:
            # move to device
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["sym"])
            loss = criterion(logits, batch_t["label"])
            bs = batch_t["label"].size(0)
            tot_loss += loss.item() * bs
            n_items += bs
            preds = logits.argmax(-1).cpu().tolist()
            y_true.extend(batch_t["label"].cpu().tolist())
            y_pred.extend(preds)
            raw_seqs.extend(batch_t["raw_seq"])
    swa = shape_weighted_accuracy(raw_seqs, y_true, y_pred)
    nwa = novelty_weighted_accuracy(raw_seqs, y_true, y_pred, train_token_set)
    return tot_loss / n_items, swa, nwa, y_true, y_pred


# ----------------------- training loop --------------------------------------
max_epochs, patience = 20, 4
best_val, wait, best_state = math.inf, 0, None

for epoch in range(1, max_epochs + 1):
    model.train()
    ep_loss = n_items = 0
    for batch in train_loader:
        # move tensors
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["sym"])
        loss = criterion(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        bs = batch_t["label"].size(0)
        ep_loss += loss.item() * bs
        n_items += bs
    train_loss = ep_loss / n_items

    val_loss, val_swa, val_nwa, y_true, y_pred = evaluate(dev_loader)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, SWA={val_swa:.4f}, NWA={val_nwa:.4f}"
    )

    exp = experiment_data["SPR_BENCH"]
    exp["losses"]["train"].append(train_loss)
    exp["losses"]["val"].append(val_loss)
    exp["metrics"]["train"].append(None)
    exp["metrics"]["val"].append({"SWA": val_swa, "NWA": val_nwa})
    exp["predictions"].append(y_pred)
    exp["ground_truth"].append(y_true)
    exp["epochs"].append(epoch)

    # early stopping
    if val_loss < best_val - 1e-4:
        best_val, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping.")
        break

# ----------------------- test ------------------------------------------------
model.load_state_dict(best_state)
test_loss, test_swa, test_nwa, y_tst, y_pst = evaluate(test_loader)
print(f"TEST: loss={test_loss:.4f}, SWA={test_swa:.4f}, NWA={test_nwa:.4f}")

# ----------------------- save ------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to working/experiment_data.npy")
