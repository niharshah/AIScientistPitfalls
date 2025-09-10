import os, pathlib, string, math, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import DatasetDict

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_acc": [],
            "val_acc": [],
            "val_fidelity": [],
            "val_fagm": [],
            "test_acc": None,
            "test_fidelity": None,
            "test_fagm": None,
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rules": {},  # class -> top character
    }
}

# ----------------------  device handling  ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------  dataset loading ----------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset loaded; train size:", len(spr["train"]))


# ----------------------  vocabulary building ------------------------------
def build_char_vocab(dataset):
    chars = set()
    for seq in dataset["sequence"]:
        chars.update(seq)
    vocab = {ch: idx for idx, ch in enumerate(sorted(list(chars)))}
    return vocab


vocab = build_char_vocab(spr["train"])
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

label2idx = {lbl: i for i, lbl in enumerate(sorted(set(spr["train"]["label"])))}
idx2label = {i: lbl for lbl, i in label2idx.items()}
num_classes = len(label2idx)
print("Number of classes:", num_classes)


# ----------------------  dataset utilities --------------------------------
def seq_to_bow(seq, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for ch in seq:
        if ch in vocab:
            vec[vocab[ch]] += 1.0
    return vec


class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, vocab, label2idx):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab
        self.label2idx = label2idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        bow = seq_to_bow(self.seqs[idx], self.vocab)
        label = self.label2idx[self.labels[idx]]
        return {
            "x": torch.tensor(bow, dtype=torch.float32),
            "y": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


batch_size = 256
train_loader = DataLoader(
    SPRTorchDataset(spr["train"], vocab, label2idx),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    SPRTorchDataset(spr["dev"], vocab, label2idx),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    SPRTorchDataset(spr["test"], vocab, label2idx),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# ----------------------  model --------------------------------------------
class LinearBagOfChar(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


model = LinearBagOfChar(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------  training utils -----------------------------------
def evaluate(model, dataloader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds, all_logits = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            loss_sum += loss.item() * batch["y"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
            all_preds.extend(preds.cpu().tolist())
            all_logits.append(logits.cpu())
    acc = correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss, np.array(all_preds)


def extract_rules(model, vocab, idx2label, top_k=1):
    # For each class, pick the character(s) with highest positive weight
    fc_weight = model.fc.weight.data.cpu().numpy()  # [C, V]
    idx2char = {idx: ch for ch, idx in vocab.items()}
    rules = {}
    for c in range(fc_weight.shape[0]):
        top_indices = fc_weight[c].argsort()[::-1][:top_k]
        top_chars = [idx2char[i] for i in top_indices]
        rules[idx2label[c]] = top_chars
    return rules


def apply_rules(rules, seq):
    # Return the predicted class or None
    matches = []
    for cls, chars in rules.items():
        for ch in chars:
            if ch in seq:
                matches.append((cls, ch))
                break
    if not matches:
        return None
    # If multiple classes matched, pick the first (could refine)
    return matches[0][0]


def compute_fidelity(model_preds, rule_preds):
    same = sum([m == r for m, r in zip(model_preds, rule_preds)])
    return same / len(model_preds)


# ----------------------  training loop ------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch["y"].size(0)
    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc, _, _ = evaluate(model, train_loader)
    val_acc, val_loss, val_preds = evaluate(model, val_loader)

    # Rule extraction & fidelity on validation
    rules = extract_rules(model, vocab, idx2label, top_k=1)
    val_rule_preds = []
    # compute rule predictions on val sets
    for seq in spr["dev"]["sequence"]:
        pred = apply_rules(rules, seq)
        val_rule_preds.append(label2idx[pred] if pred is not None else -1)

    # Convert model predictions to list of class indices
    fidelity = compute_fidelity(val_preds.tolist(), val_rule_preds)

    fagm = math.sqrt(max(val_acc, 1e-12) * max(fidelity, 1e-12))

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_fidelity"].append(fidelity)
    experiment_data["SPR_BENCH"]["metrics"]["val_fagm"].append(fagm)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

    print(
        f"Epoch {epoch}: "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} "
        f"val_acc={val_acc*100:.2f}%  fidelity={fidelity*100:.2f}%  FAGM={fagm:.4f}"
    )

# ----------------------  final test evaluation ----------------------------
test_acc, _, test_preds = evaluate(model, test_loader)
rules = extract_rules(model, vocab, idx2label, top_k=1)
test_rule_preds = [
    label2idx[apply_rules(rules, seq)] if apply_rules(rules, seq) is not None else -1
    for seq in spr["test"]["sequence"]
]
test_fidelity = compute_fidelity(test_preds.tolist(), test_rule_preds)
test_fagm = math.sqrt(max(test_acc, 1e-12) * max(test_fidelity, 1e-12))

experiment_data["SPR_BENCH"]["metrics"]["test_acc"] = test_acc
experiment_data["SPR_BENCH"]["metrics"]["test_fidelity"] = test_fidelity
experiment_data["SPR_BENCH"]["metrics"]["test_fagm"] = test_fagm
experiment_data["SPR_BENCH"]["rules"] = rules
experiment_data["SPR_BENCH"]["predictions"] = test_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = [
    label2idx[lbl] for lbl in spr["test"]["label"]
]

print(
    f"\nTEST RESULTS  |  acc={test_acc*100:.2f}%  fidelity={test_fidelity*100:.2f}%  FAGM={test_fagm:.4f}"
)

# ----------------------  save everything ----------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
with open(os.path.join(working_dir, "rules.json"), "w") as f:
    json.dump(rules, f, indent=2)
