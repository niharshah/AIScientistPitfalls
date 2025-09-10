import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim, pathlib
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import defaultdict

# ------------------- WORK DIR --------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- DEVICE ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- SEED ------------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ------------------- DATA ------------------------------
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


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
dsets = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in dsets.items()})

# build char vocab
chars = set()
for s in dsets["train"]["sequence"]:
    chars.update(list(s))
char2id = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0 = PAD
vocab_size = len(char2id) + 1
print("vocab size", vocab_size)


def encode(seq: str, max_len: int):
    ids = [char2id[c] for c in seq]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array(ids, dtype=np.int64)


max_len = max(len(s) for s in dsets["train"]["sequence"])
print("max seq len", max_len)


def build_xy(split):
    X = np.stack([encode(s, max_len) for s in dsets[split]["sequence"]])
    labels = (
        dsets["train"].features["label"].names
        if hasattr(dsets["train"].features["label"], "names")
        else sorted(set(dsets["train"]["label"]))
    )
    lab2id = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2id[l] for l in dsets[split]["label"]], dtype=np.int64)
    return X, y, labels


X_train, y_train, labels = build_xy("train")
X_dev, y_dev, _ = build_xy("dev")
X_test, y_test, _ = build_xy("test")
num_classes = len(labels)


class SPRDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X[idx]), "y": torch.tensor(self.y[idx])}


batch_size = 128
train_loader = DataLoader(
    SPRDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRDataset(X_dev, y_dev), batch_size=batch_size)
test_loader = DataLoader(SPRDataset(X_test, y_test), batch_size=batch_size)


# ------------------- MODEL -----------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab, emb_dim, num_classes, kernels=(2, 3, 4), num_filters=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernels]
        )
        self.fc = nn.Linear(num_filters * len(kernels), num_classes)

    def forward(self, x, return_feat=False):
        # x: (B,L)
        e = self.embed(x).transpose(1, 2)  # (B,emb,L)
        feats = [torch.relu(conv(e)).max(dim=2)[0] for conv in self.convs]  # list (B,F)
        feat = torch.cat(feats, 1)  # (B, F*K)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


# ------------------- EXPERIMENT TRACK ------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": [], "rule_fidelity": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_test.tolist(),
        "top_rules": defaultdict(list),
    }
}

# ------------------- TRAIN / EVAL FUNCS ----------------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, collect_feat=False):
    model.eval()
    tot, correct, ls = 0, 0, 0.0
    feats_list = []
    logits_list = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, feat = model(batch["x"], return_feat=True)
            loss = criterion(logits, batch["y"])
            ls += loss.item() * batch["y"].size(0)
            pred = logits.argmax(1)
            correct += (pred == batch["y"]).sum().item()
            tot += batch["y"].size(0)
            if collect_feat:
                feats_list.append(feat.cpu())
                logits_list.append(logits.cpu())
    acc = correct / tot
    loss = ls / tot
    if collect_feat:
        return acc, loss, torch.cat(feats_list), torch.cat(logits_list)
    return acc, loss, None, None


# ------------------- TRAINING LOOP ---------------------
lr_choices = [1e-3, 3e-4]
epochs = 10
best_dev = -1
top_k = 10

for lr in lr_choices:
    print(f"\n==== LR {lr} ====")
    model = CharCNN(vocab_size, 16, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    l1_lambda = 1e-4
    run_train_acc, run_val_acc, run_rfa, run_train_loss, run_val_loss = (
        [],
        [],
        [],
        [],
        [],
    )
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, ls = 0, 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            # L1 sparsity on classifier weights
            l1_pen = model.fc.weight.abs().sum() * l1_lambda
            (loss + l1_pen).backward()
            optimizer.step()
            ls += loss.item() * batch["y"].size(0)
            pred = logits.argmax(1)
            correct += (pred == batch["y"]).sum().item()
            tot += batch["y"].size(0)
        train_acc = correct / tot
        train_loss = ls / tot
        val_acc, val_loss, dev_feat, dev_logits = evaluate(
            model, dev_loader, collect_feat=True
        )

        # Rule Fidelity: keep top_k filters per class
        W = model.fc.weight.detach().cpu().numpy()  # (C,F)
        b = model.fc.bias.detach().cpu().numpy()
        W_trunc = np.zeros_like(W)
        for c in range(num_classes):
            idxs = np.argsort(-np.abs(W[c]))[:top_k]
            W_trunc[c, idxs] = W[c, idxs]
        logits_full = dev_feat @ W.T + b
        logits_trunc = dev_feat @ W_trunc.T + b
        rfa = (logits_full.argmax(1) == logits_trunc.argmax(1)).float().mean().item()

        run_train_acc.append(train_acc)
        run_val_acc.append(val_acc)
        run_rfa.append(rfa)
        run_train_loss.append(train_loss)
        run_val_loss.append(val_loss)
        print(
            f"Epoch {ep}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} RFA={rfa:.3f}"
        )

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(run_train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(run_val_acc)
    experiment_data["SPR_BENCH"]["metrics"]["rule_fidelity"].append(run_rfa)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(run_train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(run_val_loss)

    if val_acc > best_dev:
        best_dev = val_acc
        # save predictions on test
        test_acc, _, test_feat, _ = evaluate(model, test_loader)
        print(f"*** new best dev_acc={val_acc:.3f}, test_acc={test_acc:.3f}")
        best_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["x"])
                best_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        experiment_data["SPR_BENCH"]["predictions"] = best_preds

        # rule extraction: top n-grams activating each retained filter
        # gather global activations to map filter -> substrings
        activ_dict = defaultdict(list)  # filter idx -> (score, substring)
        model.eval()
        with torch.no_grad():
            for seq in dsets["train"]["sequence"][:5000]:  # sample subset for speed
                ids = torch.tensor(encode(seq, max_len)).unsqueeze(0).to(device)
                emb = model.embed(ids).transpose(1, 2)
                for conv_i, conv in enumerate(model.convs):
                    act = torch.relu(conv(emb)).squeeze(0)  # (F, L-k+1)
                    for f in range(act.size(0)):
                        m, val = act[f].max(0)
                        activ_dict[conv_i * conv.out_channels + f].append(
                            (
                                m.item(),
                                seq[val.item() : val.item() + conv.kernel_size[0]],
                            )
                        )
        # keep top substrings
        for filt in range(model.fc.weight.size(1)):
            if any(
                filt in np.argsort(-np.abs(W[c]))[:top_k] for c in range(num_classes)
            ):
                top = sorted(activ_dict[filt], key=lambda x: -x[0])[:3]
                experiment_data["SPR_BENCH"]["top_rules"][filt] = [s for _, s in top]

# ------------------- SAVE ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("saved experiment_data.npy")
