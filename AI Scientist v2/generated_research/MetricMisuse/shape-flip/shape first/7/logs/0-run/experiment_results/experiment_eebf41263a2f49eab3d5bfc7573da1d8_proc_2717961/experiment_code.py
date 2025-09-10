import os, random, time, json, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------- misc utils ---------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print("Using device:", device)


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1e-6)


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    return 2 * swa * cwa / max(swa + cwa, 1e-6)


# --------------------------- data ---------------------------
def load_spr_bench(root_path: str) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root_path, csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


def make_synthetic_dataset(path: str, n_train=2000, n_dev=500, n_test=500):
    shapes, colors = list("STCH"), list("RGBY")

    def rand_seq():
        toks = [
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(3, 10))
        ]
        return " ".join(toks)

    def rule(seq):
        return int(count_shape_variety(seq) > count_color_variety(seq))

    os.makedirs(path, exist_ok=True)

    def dump(n, fname):
        rows = ["id,sequence,label"]
        for i in range(n):
            s = rand_seq()
            rows.append(f"{i},{s},{rule(s)}")
        with open(os.path.join(path, fname), "w") as f:
            f.write("\n".join(rows))

    dump(n_train, "train.csv")
    dump(n_dev, "dev.csv")
    dump(n_test, "test.csv")


root = os.getenv("SPR_PATH", "SPR_BENCH")
if not (
    os.path.exists(root)
    and all(
        os.path.exists(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("Dataset not found, generating synthetic.")
    make_synthetic_dataset(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# --------------------------- vocab / encode ---------------------------
def build_vocab(dataset):
    v = {"<pad>": 0, "<unk>": 1}
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in v:
                v[tok] = len(v)
    return v


vocab = build_vocab(spr["train"])
max_len = 20
num_labels = len(set(spr["train"]["label"]))


def encode(seq, v=vocab, max_len=max_len):
    ids = [v.get(tok, v["<unk>"]) for tok in seq.split()][:max_len]
    ids += [v["<pad>"]] * (max_len - len(ids))
    return ids


class SPRTorch(Dataset):
    def __init__(self, hf):
        self.seq = hf["sequence"]
        self.y = hf["label"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
            "raw": self.seq[idx],
        }


batch_size = 64
train_ds, dev_ds, test_ds = (
    SPRTorch(spr["train"]),
    SPRTorch(spr["dev"]),
    SPRTorch(spr["test"]),
)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)


# --------------------------- model def ---------------------------
class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        pooled = out.mean(1)
        return self.fc(pooled)


# --------------------------- train / eval helpers ---------------------------
def run_epoch(model, dl, criterion, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    tot_loss = 0
    y_true = []
    y_pred = []
    seqs = []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"])
        loss = criterion(logits, batch["label"])
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    avg_loss = tot_loss / len(dl.dataset)
    swa = shape_weighted_accuracy(seqs, y_true, y_pred)
    cwa = color_weighted_accuracy(seqs, y_true, y_pred)
    hwa = harmonic_weighted_accuracy(seqs, y_true, y_pred)
    return avg_loss, swa, cwa, hwa, y_true, y_pred


# --------------------------- hyperparam sweep ---------------------------
hidden_sizes = [32, 64, 128, 256]
epochs = 5
experiment_data = {"gru_hidden_size": {}}
best_val_hwa, best_key = -1, None

for hid in hidden_sizes:
    key = f"hid_{hid}"
    print(f"\n=== training model with hidden size {hid} ===")
    model = BaselineGRU(len(vocab), hid=hid, num_classes=num_labels).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    experiment_data["gru_hidden_size"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, epochs + 1):
        tr_loss, _, _, tr_hwa, _, _ = run_epoch(model, train_dl, crit, opt)
        val_loss, val_swa, val_cwa, val_hwa, _, _ = run_epoch(model, dev_dl, crit)
        experiment_data["gru_hidden_size"][key]["losses"]["train"].append(
            (epoch, tr_loss)
        )
        experiment_data["gru_hidden_size"][key]["losses"]["val"].append(
            (epoch, val_loss)
        )
        experiment_data["gru_hidden_size"][key]["metrics"]["train"].append(
            (epoch, tr_hwa)
        )
        experiment_data["gru_hidden_size"][key]["metrics"]["val"].append(
            (epoch, val_hwa)
        )
        print(f"Epoch {epoch}: val_loss {val_loss:.4f} HWA {val_hwa:.4f}")
    # test evaluation
    _, _, _, test_hwa, test_y, test_pred = run_epoch(model, test_dl, crit)
    experiment_data["gru_hidden_size"][key]["predictions"] = test_pred
    experiment_data["gru_hidden_size"][key]["ground_truth"] = test_y
    experiment_data["gru_hidden_size"][key]["test_hwa"] = test_hwa
    print(f"Hidden {hid}: Test HWA {test_hwa:.4f}")
    if val_hwa > best_val_hwa:
        best_val_hwa, val_best, test_best = val_hwa, key, test_hwa

print(
    f"\nBest hidden size based on dev: {best_key} (dev HWA {best_val_hwa:.4f}, test HWA {test_best:.4f})"
)

# --------------------------- save ---------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
