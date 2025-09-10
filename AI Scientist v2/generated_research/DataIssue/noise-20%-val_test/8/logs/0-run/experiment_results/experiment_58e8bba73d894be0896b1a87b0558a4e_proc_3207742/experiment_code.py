import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------
# -------------------- data helper -----------------------------------
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


def obtain_dataset() -> DatasetDict:
    root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    try:
        d = load_spr_bench(root)
        print("Loaded real SPR_BENCH.")
        return d
    except Exception:
        print("Generating synthetic parity dataset.")
        rng = np.random.default_rng(0)
        vocab = list("ABCDEFGHIJ")

        def gen(n):
            seq, lab = [], []
            for i in range(n):
                ln = rng.integers(5, 12)
                s = "".join(rng.choice(vocab, size=ln))
                seq.append(s)
                lab.append(int(s.count("A") % 2 == 0))
            return HFDataset.from_dict(
                {"id": list(range(n)), "sequence": seq, "label": lab}
            )

        return DatasetDict(train=gen(2000), dev=gen(500), test=gen(500))


dsets = obtain_dataset()

# --------------- vocabulary & tensorisation -------------------------
chars = sorted({c for split in dsets for seq in dsets[split]["sequence"] for c in seq})
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 for PAD
vocab_size = len(char2idx) + 1
max_len = max(len(s) for s in dsets["train"]["sequence"])  # for padding


def encode(seq):
    idxs = [char2idx[c] for c in seq]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return np.array(idxs, dtype=np.int64)


class SPRTorchDataset(Dataset):
    def __init__(self, split):
        self.X = [encode(s) for s in dsets[split]["sequence"]]
        self.y = dsets[split]["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.X[i]),
            "labels": torch.tensor(self.y[i], dtype=torch.long),
        }


batch_size = 128
loaders = {
    sp: DataLoader(SPRTorchDataset(sp), batch_size=batch_size, shuffle=(sp == "train"))
    for sp in ["train", "dev", "test"]
}


# -------------------- BiGRU model -----------------------------------
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab, emb_dim=32, hid=32, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        hcat = torch.cat([h[0], h[1]], dim=1)
        return self.fc(hcat), hcat.detach()


model = BiGRUClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- containers ----------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "IRF": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ----------------------- training loop ------------------------------
def run_epoch(split, train_flag=True):
    loader = loaders[split]
    total_loss, total_correct, total = 0, 0, 0
    latent_list, labels_list = [], []
    if train_flag:
        model.train()
    else:
        model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out, latent = model(batch["input_ids"])
        loss = criterion(out, batch["labels"])
        if train_flag:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * len(batch["labels"])
        preds = out.argmax(1)
        total_correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])
        latent_list.append(latent.cpu().numpy())
        labels_list.append(preds.cpu().numpy())
    return (
        total_loss / total,
        total_correct / total,
        np.concatenate(latent_list),
        np.concatenate(labels_list),
    )


num_epochs = 5
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_latent, tr_pred = run_epoch("train", True)
    val_loss, val_acc, val_latent, val_pred = run_epoch("dev", False)
    # fit decision tree on train latent to imitate model
    dt = DecisionTreeClassifier(max_depth=5, random_state=0)
    dt.fit(tr_latent, tr_pred)
    irf = (dt.predict(val_latent) == val_pred).mean()
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc={val_acc:.3f}, IRF={irf:.3f}"
    )
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append(tr_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["IRF"].append(irf)

# ------------------ final evaluation & rule extraction --------------
test_loss, test_acc, test_latent, test_pred = run_epoch("test", False)
dt_final = DecisionTreeClassifier(max_depth=5, random_state=0)
dt_final.fit(tr_latent, tr_pred)
test_irf = (dt_final.predict(test_latent) == test_pred).mean()
print(f"Test accuracy={test_acc:.3f}, Test IRF={test_irf:.3f}")


# rule extraction
def extract_rules(tree, feat_dim):
    rules = []
    tree_ = tree.tree_

    def rec(node, conds):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            thr = tree_.threshold[node]
            feat = tree_.feature[node]
            rec(tree_.children_left[node], conds + [f"f{feat}<= {thr:.3f}"])
            rec(tree_.children_right[node], conds + [f"f{feat}> {thr:.3f}"])
        else:
            pred = np.argmax(tree_.value[node][0])
            rule = " AND ".join(conds) if conds else "TRUE"
            rules.append(f"IF {rule} THEN label={pred}")

    rec(0, [])
    return rules


rules = extract_rules(dt_final, tr_latent.shape[1])
with open(os.path.join(working_dir, "extracted_rules.txt"), "w") as f:
    f.write("\n".join(rules))
print(f"Saved {len(rules)} rules.")

# ---------------- confusion matrix & saving -------------------------
cm = confusion_matrix([ex["label"] for ex in dsets["test"]], test_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Pred")
ax.set_ylabel("True")
ax.set_title("Confusion")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
plt.colorbar(im, ax=ax)
plt.savefig(os.path.join(working_dir, "confusion_matrix.png"))
plt.close()

experiment_data["SPR_BENCH"]["predictions"] = test_pred.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = [ex["label"] for ex in dsets["test"]]
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
