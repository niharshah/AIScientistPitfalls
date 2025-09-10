import os, pathlib, random, string, time, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ------------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------------
# GPU / CPU handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------------
# some helper symbolic functions (from baseline)
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(seq) for seq in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------------------------------------------------------------
# fallback synthetic generator in case real data unavailable
def random_token():
    return random.choice(string.ascii_uppercase[:10]) + random.choice(string.digits[:5])


def generate_synthetic_split(n_rows, seed):
    random.seed(seed)
    seqs, labels = [], []
    for i in range(n_rows):
        length = random.randint(3, 8)
        seq = " ".join(random_token() for _ in range(length))
        # simple rule: 1 if #shapes == #colours else 0
        labels.append(int(count_shape_variety(seq) == count_color_variety(seq)))
        seqs.append(seq)
    return {"id": list(range(n_rows)), "sequence": seqs, "label": labels}


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    if root.exists():

        def _load(csv_name):
            return load_dataset("csv", data_files=str(root / csv_name), split="train")

        print(f"Loading SPR_BENCH from {root}")
        return DatasetDict(
            train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
        )
    print("SPR_BENCH folder not found; using synthetic data")
    return DatasetDict(
        train=HFDataset.from_dict(generate_synthetic_split(2000, 1)),
        dev=HFDataset.from_dict(generate_synthetic_split(500, 2)),
        test=HFDataset.from_dict(generate_synthetic_split(1000, 3)),
    )


DATA_PATH = pathlib.Path("./SPR_BENCH")
dsets = load_spr_bench(DATA_PATH)

# ------------------------------------------------------------------------
# tokenisation helpers
shape_vocab = {ch: i for i, ch in enumerate(string.ascii_uppercase[:26])}
colour_vocab = {d: i for i, d in enumerate(string.digits[:10])}
PAD_IDX = (
    0  # for padding sequences (actually never used because variable len aggregator)
)


def split_tokens(seq):
    return seq.strip().split()


class SPRDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        toks = split_tokens(seq)
        shapes = torch.tensor([shape_vocab[t[0]] for t in toks], dtype=torch.long)
        colours = torch.tensor([colour_vocab[t[1]] for t in toks], dtype=torch.long)
        # symbolic features
        sym_features = torch.tensor(
            [len(toks), count_shape_variety(seq), count_color_variety(seq)],
            dtype=torch.float32,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "shapes": shapes,
            "colours": colours,
            "sym": sym_features,
            "label": label,
            "raw_seq": seq,
        }


def collate(batch):
    # variable length: keep lists, process inside model
    shapes = [item["shapes"] for item in batch]
    colours = [item["colours"] for item in batch]
    sym = torch.stack([item["sym"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    raw_seqs = [item["raw_seq"] for item in batch]
    return {
        "shapes": shapes,
        "colours": colours,
        "sym": sym,
        "label": labels,
        "raw_seq": raw_seqs,
    }


train_loader = DataLoader(
    SPRDataset(dsets["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(dsets["test"]), batch_size=256, shuffle=False, collate_fn=collate
)

num_classes = len(set(dsets["train"]["label"]))


# ------------------------------------------------------------------------
# neural-symbolic model
class DeepSetNS(nn.Module):
    def __init__(self, d_embed=32, sym_dim=3, hidden=64, n_classes=2):
        super().__init__()
        self.shape_embed = nn.Embedding(len(shape_vocab), d_embed)
        self.colour_embed = nn.Embedding(len(colour_vocab), d_embed)
        self.ff = nn.Sequential(nn.Linear(d_embed, d_embed), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(d_embed + sym_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, shapes_list, colours_list, sym):
        batch_vecs = []
        for sh, co in zip(shapes_list, colours_list):
            # move to device
            sh = sh.to(device)
            co = co.to(device)
            tok_vecs = self.shape_embed(sh) + self.colour_embed(co)  # [seq_len, d]
            tok_vecs = self.ff(tok_vecs)
            seq_vec = tok_vecs.mean(dim=0)  # DeepSets aggregation
            batch_vecs.append(seq_vec)
        batch_vecs = torch.stack(batch_vecs)  # [B, d_embed]
        x = torch.cat([batch_vecs, sym.to(device)], dim=1)
        logits = self.classifier(x)
        return logits


model = DeepSetNS(n_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------------------
# experiment tracking structure
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_swa": [], "val_swa": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": dsets["test"]["label"],
        "epochs": [],
    }
}


def evaluate(loader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["shapes"], batch["colours"], batch["sym"])
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_seqs.extend(batch["raw_seq"])
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    loss = 0.0  # optional
    return swa, all_preds


EPOCHS = 15
best_val_swa = -1
best_state = None

for epoch in range(1, EPOCHS + 1):
    # ---------- training ----------
    model.train()
    running_loss = 0.0
    running_seqs, running_preds, running_labels = [], [], []
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["shapes"], batch["colours"], batch["sym"])
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        running_preds.extend(preds)
        running_labels.extend(batch["label"].numpy())
        running_seqs.extend(batch["raw_seq"])
    train_loss = running_loss / len(train_loader.dataset)
    train_swa = shape_weighted_accuracy(running_seqs, running_labels, running_preds)

    # ---------- validation ----------
    val_swa, _ = evaluate(dev_loader)

    print(
        f"Epoch {epoch}: validation_loss = {train_loss:.4f} | Train SWA {train_swa:.3f} | Val SWA {val_swa:.3f}"
    )

    experiment_data["SPR_BENCH"]["metrics"]["train_swa"].append(train_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_swa"].append(val_swa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    # keep best
    if val_swa > best_val_swa:
        best_val_swa = val_swa
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------------------------------------------------------------------
# test evaluation with best model
model.load_state_dict(best_state)
test_swa, test_preds = evaluate(test_loader)
print(f"\nBest Val SWA = {best_val_swa:.3f} | Test SWA = {test_swa:.3f}")

experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["metrics"]["test_swa"] = test_swa

# ------------------------------------------------------------------------
# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
