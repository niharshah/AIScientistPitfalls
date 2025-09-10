import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, time, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, log_loss

# --------------------------- DEVICE ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------- DATA LOADING --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def get_dataset():
    try:
        ds_root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
        dset = load_spr_bench(ds_root)
        print("Loaded SPR_BENCH")
        return dset
    except Exception as e:
        print("SPR_BENCH unavailable, generating synthetic data.")
        rng = np.random.default_rng(0)
        vocab = list("ABC")

        def gen(n):
            seqs, labels = [], []
            for i in range(n):
                l = rng.integers(4, 10)
                s = "".join(rng.choice(vocab, size=l))
                labels.append(int(s.count("A") % 2 == 0))
                seqs.append(s)
            return Dataset.from_dict(
                {"id": list(range(n)), "sequence": seqs, "label": labels}
            )

        return DatasetDict(train=gen(600), dev=gen(200), test=gen(200))


dsets = get_dataset()

# -------------------- TOKENISATION & VECTORS ------------------------
# build vocabulary
chars = sorted({c for split in dsets for seq in dsets[split]["sequence"] for c in seq})
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # reserve 0 for pad
vocab_size = len(char2idx) + 1
print("Vocab:", chars)


def encode(seq):
    return [char2idx[c] for c in seq]


for split in dsets:
    dsets[split] = dsets[split].map(lambda x: {"input_ids": encode(x["sequence"])})


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["input_ids"]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {"input_ids": ids.to(device), "labels": labels.to(device)}


train_loader = DataLoader(
    dsets["train"], batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dsets["dev"], batch_size=256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(
    dsets["test"], batch_size=256, shuffle=False, collate_fn=collate
)


# -------------------------- MODEL -----------------------------------
class CharGRUClassifier(nn.Module):
    def __init__(self, vocab, emb=32, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)


model = CharGRUClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------- TRAINING LOOP & METRICS ------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "IRF": {"val": [], "test": []},
        "predictions": [],
        "ground_truth": dsets["test"]["label"],
    }
}


def evaluate(loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"])
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = log_loss(labels.numpy(), torch.softmax(logits, 1).numpy())
    acc = (logits.argmax(1) == labels).float().mean().item()
    return loss, acc, logits.argmax(1).numpy()


best_val_loss = 1e9
patience, wait = 2, 0
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    running_acc = 0
    total = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch["labels"])
        running_acc += (logits.argmax(1) == batch["labels"]).float().sum().item()
        total += len(batch["labels"])
    train_loss = running_loss / total
    train_acc = running_acc / total

    val_loss, val_acc, _ = evaluate(dev_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        break

model.load_state_dict(torch.load(os.path.join(working_dir, "best_model.pt")))


# -------------- SURROGATE DECISION TREE & IRF -----------------------
def bag_of_chars(seq):
    v = np.zeros(len(chars), dtype=np.float32)
    for ch in seq:
        v[char2idx[ch] - 1] += 1
    return v


X_train = np.stack([bag_of_chars(s) for s in dsets["train"]["sequence"]])
with torch.no_grad():
    y_train_model = np.concatenate(
        [
            model(collate([d])["input_ids"]).argmax(1).cpu().numpy()
            for d in dsets["train"]
        ]
    )
tree = DecisionTreeClassifier(max_depth=6, random_state=0)
tree.fit(X_train, y_train_model)


def eval_irf(split_name, loader):
    X = np.stack([bag_of_chars(s) for s in dsets[split_name]["sequence"]])
    surrogate_pred = tree.predict(X)
    _, _, model_pred = evaluate(loader)
    return (surrogate_pred == model_pred).mean()


irf_val = eval_irf("dev", dev_loader)
irf_test = eval_irf("test", test_loader)
experiment_data["SPR_BENCH"]["IRF"]["val"].append(irf_val)
experiment_data["SPR_BENCH"]["IRF"]["test"].append(irf_test)
print(f"IRF dev={irf_val:.3f}, IRF test={irf_test:.3f}")

# --------------------- TEST ACCURACY --------------------------------
test_loss, test_acc, test_pred = evaluate(test_loader)
experiment_data["SPR_BENCH"]["metrics"]["test"] = [test_acc]
experiment_data["SPR_BENCH"]["predictions"] = test_pred.tolist()
print(f"Test accuracy = {test_acc:.4f}")


# ------------------- RULE STRING EXTRACTION -------------------------
def tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            thr = tree_.threshold[node]
            recurse(tree_.children_left[node], path + [f"{name}<= {thr:.1f}"])
            recurse(tree_.children_right[node], path + [f"{name}> {thr:.1f}"])
        else:
            val = np.argmax(tree_.value[node][0])
            rule = " AND ".join(path) if path else "TRUE"
            rules.append(f"IF {rule} THEN label={val}")

    recurse(0, [])
    return rules


rule_strings = tree_to_rules(tree, chars)
with open(os.path.join(working_dir, "extracted_rules.txt"), "w") as f:
    f.write("\n".join(rule_strings))
print(f"Wrote {len(rule_strings)} rules to extracted_rules.txt")

# ----------------------- SAVE METRICS -------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to experiment_data.npy")
