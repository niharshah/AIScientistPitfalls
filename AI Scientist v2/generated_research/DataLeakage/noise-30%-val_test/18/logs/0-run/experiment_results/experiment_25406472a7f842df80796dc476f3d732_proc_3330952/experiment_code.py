import os, pathlib, random, string, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------------------- reproducibility -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- device & dir ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------------- load SPR_BENCH --------------------------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset, DatasetDict

    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


data_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if data_path.exists():
    dsets = load_spr_bench(data_path)
else:  # ------- tiny synthetic fallback -------------
    from datasets import Dataset, DatasetDict

    def synth_split(n):
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(5, 15)
            seq = "".join(
                random.choices(list(string.ascii_lowercase) + ["#", "@", "&"], k=L)
            )
            labels.append(int(seq.count("#") % 2 == 0))
            seqs.append(seq)
        return Dataset.from_dict(
            {"id": list(range(n)), "sequence": seqs, "label": labels}
        )

    dsets = DatasetDict()
    dsets["train"], dsets["dev"], dsets["test"] = (
        synth_split(512),
        synth_split(128),
        synth_split(128),
    )
print({k: len(v) for k, v in dsets.items()})

# ---------------------- vocab & encoding ------------------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for seq in dsets["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(ch, vocab[UNK]) for ch in seq]


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=["sequence"]
    )


# ---------------------- DataLoader ------------------------------
def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), vocab[PAD], dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": padded, "labels": labels}


batch_size = 128
train_loader = DataLoader(
    dsets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    dsets["dev"], batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    dsets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------------------- Model definition ------------------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        em = self.emb(x)
        out, _ = self.lstm(em)
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)


# ---------------------- experiment data dict --------------------
experiment_data = {"hidden_size": {"SPR_BENCH": {}}}  # Filled per hidden size


# ---------------------- training / evaluation -------------------
def evaluate(model, loader, criterion):
    model.eval()
    losses, preds, gts = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            losses.append(criterion(logits, batch["labels"]).item())
            preds.extend(logits.argmax(1).cpu().numpy())
            gts.extend(batch["labels"].cpu().numpy())
    return np.mean(losses), f1_score(gts, preds, average="macro"), preds, gts


# ---------------------- hyperparameter sweep --------------------
hidden_sizes = [64, 128, 256, 512]
epochs = 5
for hs in hidden_sizes:
    run_key = f"hs_{hs}"
    print(f"\n=== Training with hidden size {hs} ===")
    exp_run = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "best_dev_f1": 0.0,
        "test_f1": 0.0,
    }
    model = SPRClassifier(vocab_size, hidden=hs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses, train_preds, train_gts = [], [], []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_gts.extend(batch["labels"].cpu().numpy())
        train_f1 = f1_score(train_gts, train_preds, average="macro")

        # ---- validation ----
        val_loss, val_f1, val_preds, val_gts = evaluate(model, dev_loader, criterion)

        # ---- logging ----
        print(
            f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}, val_F1={val_f1:.4f}"
        )
        exp_run["metrics"]["train"].append(train_f1)
        exp_run["metrics"]["val"].append(val_f1)
        exp_run["losses"]["train"].append(np.mean(train_losses))
        exp_run["losses"]["val"].append(val_loss)
        exp_run["epochs"].append(epoch)

        # ---- save best ----
        if val_f1 > exp_run["best_dev_f1"]:
            exp_run["best_dev_f1"] = val_f1
            exp_run["predictions"] = val_preds
            exp_run["ground_truth"] = val_gts
            best_state = model.state_dict()

    # ---- reload best and evaluate on test ----
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_f1, _, _ = evaluate(model, test_loader, criterion)
    exp_run["test_f1"] = test_f1
    print(
        f"Finished hidden size {hs}: Best Dev F1 = {exp_run['best_dev_f1']:.4f} | Test F1 = {test_f1:.4f}"
    )

    # store
    experiment_data["hidden_size"]["SPR_BENCH"][run_key] = exp_run

# ---------------------- save everything -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
