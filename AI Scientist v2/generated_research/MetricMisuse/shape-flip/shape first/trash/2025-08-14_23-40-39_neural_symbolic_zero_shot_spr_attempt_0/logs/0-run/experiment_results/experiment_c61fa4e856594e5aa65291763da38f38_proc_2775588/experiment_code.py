import os, math, pathlib, numpy as np, torch, random
from collections import Counter
from datetime import datetime
from datasets import load_dataset, DatasetDict, disable_caching
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Repro -----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ----------------- Save container -----------------
experiment_data = {"weight_decay": {}}

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Disable HF cache (keeps workspace clean) -------------
disable_caching()


# ----------------- SPR path resolver -----------------
def resolve_spr_path() -> pathlib.Path:
    env_path = os.getenv("SPR_PATH")
    if env_path and (pathlib.Path(env_path) / "train.csv").exists():
        return pathlib.Path(env_path)
    cur = pathlib.Path.cwd()
    for parent in [cur] + list(cur.parents):
        cand = parent / "SPR_BENCH"
        if (cand / "train.csv").exists():
            return cand
    fb = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if (fb / "train.csv").exists():
        return fb
    raise FileNotFoundError("SPR_BENCH not found")


# ----------------- Load dataset -----------------
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


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def swa_metric(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def cwa_metric(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ----------------- Hyperparams -----------------
EMB_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, LR = 64, 128, 128, 5, 1e-3
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

# ----------------- Data & vocab -----------------
DATA_PATH = resolve_spr_path()
spr = load_spr_bench(DATA_PATH)
token_counter = Counter(tok for seq in spr["train"]["sequence"] for tok in seq.split())
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab.update({tok: i + 2 for i, tok in enumerate(token_counter)})
inv_vocab = {i: t for t, i in vocab.items()}
label_set = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(label_set)}
id2label = {i: l for l, i in label2id.items()}


def encode(seq):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs, self.labels = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
            "seq_str": self.seqs[idx],
        }


def collate(batch):
    lens = [len(x["input_ids"]) for x in batch]
    max_len = max(lens)
    input_ids = torch.full((len(batch), max_len), vocab[PAD_TOKEN], dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "seq_strs": [b["seq_str"] for b in batch],
        "lengths": torch.tensor(lens),
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)


# ----------------- Model -----------------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, out_dim)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        emb = self.emb(input_ids)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        x = self.relu(self.fc1(avg))
        return self.fc2(x)


criterion = nn.CrossEntropyLoss()


def evaluate(loader, model):
    model.eval()
    tot_loss, n = 0, 0
    preds, labels, seqs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            p = logits.argmax(1).cpu().tolist()
            l = batch["labels"].cpu().tolist()
            preds.extend(p)
            labels.extend(l)
            seqs.extend(batch["seq_strs"])
    loss = tot_loss / max(n, 1)
    swa, cwa = swa_metric(seqs, labels, preds), cwa_metric(seqs, labels, preds)
    bps = math.sqrt(swa * cwa)
    return loss, swa, cwa, bps, preds, labels


# ----------------- Hyper-parameter sweep -----------------
weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
for wd in weight_decays:
    key = str(wd)
    experiment_data["weight_decay"][key] = {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "val_swa": [],
            "val_cwa": [],
            "val_bps": [],
        },
        "predictions": {"dev": [], "test": []},
        "ground_truth": {"dev": [], "test": []},
        "timestamps": [],
    }
    model = SPRClassifier(len(vocab), EMB_DIM, len(label_set)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    best_bps = -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0, 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            running += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)
        train_loss = running / seen
        val_loss, swa, cwa, bps, _, _ = evaluate(dev_loader, model)
        print(
            f"wd={wd} | epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | BPS {bps:.4f}"
        )
        exp = experiment_data["weight_decay"][key]["metrics"]
        exp["train_loss"].append(train_loss)
        exp["val_loss"].append(val_loss)
        exp["val_swa"].append(swa)
        exp["val_cwa"].append(cwa)
        exp["val_bps"].append(bps)
        experiment_data["weight_decay"][key]["timestamps"].append(
            datetime.utcnow().isoformat()
        )
        if bps > best_bps:
            best_state = model.state_dict()
            best_bps = bps
    # load best state for final eval
    model.load_state_dict(best_state)
    dev_loss, dev_swa, dev_cwa, dev_bps, dev_pred, dev_gt = evaluate(dev_loader, model)
    test_loss, test_swa, test_cwa, test_bps, test_pred, test_gt = evaluate(
        test_loader, model
    )
    experiment_data["weight_decay"][key]["predictions"]["dev"] = dev_pred
    experiment_data["weight_decay"][key]["ground_truth"]["dev"] = dev_gt
    experiment_data["weight_decay"][key]["predictions"]["test"] = test_pred
    experiment_data["weight_decay"][key]["ground_truth"]["test"] = test_gt
    print(f"==wd {wd} FINAL== dev BPS {dev_bps:.4f} | test BPS {test_bps:.4f}")

# ----------------- Save -----------------
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", "experiment_data.npy"), experiment_data)
