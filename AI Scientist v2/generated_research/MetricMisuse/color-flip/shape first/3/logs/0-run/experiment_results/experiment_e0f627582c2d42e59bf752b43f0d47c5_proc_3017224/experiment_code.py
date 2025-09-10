import os, random, pathlib, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------  basics & utils  -----------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _ld(csv_name):
            return load_dataset(
                "csv",
                data_files=str(root / csv_name),
                split="train",
                cache_dir=".cache_dsets",
            )

        return True, {
            sp.split(".")[0]: _ld(sp) for sp in ["train.csv", "dev.csv", "test.csv"]
        }
    except Exception as e:
        print("Could not load SPR_BENCH, falling back to synthetic data.", e)
        return False, {}


def make_synth_dataset(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(3, 10)
        s = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(s)
        labels.append(int(count_shape_variety(s) > count_color_variety(s)))
    return {"sequence": seqs, "label": labels}


class SPRDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.seqs, self.labels, self.vocab, self.max_len = (
            sequences,
            labels,
            vocab,
            max_len,
        )

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in seq.split()]
        ids = ids[: self.max_len] + [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h.squeeze(0))


# ------------------------------------  data prep  ----------------------------------------------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    train_dict = {"sequence": raw["train"]["sequence"], "label": raw["train"]["label"]}
    dev_dict = {"sequence": raw["dev"]["sequence"], "label": raw["dev"]["label"]}
    test_dict = {"sequence": raw["test"]["sequence"], "label": raw["test"]["label"]}
else:
    train_dict, dev_dict, test_dict = (
        make_synth_dataset(2000),
        make_synth_dataset(400),
        make_synth_dataset(400),
    )

all_tokens = set(tok for seq in train_dict["sequence"] for tok in seq.split())
vocab = {tok: i + 2 for i, tok in enumerate(sorted(all_tokens))}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
pad_idx = vocab["<pad>"]
max_len = max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -------------------------------  experiment bookkeeping  --------------------------------------
experiment_data = {"batch_size_tuning": {"SPR_BENCH": {}}}


# -----------------------------------  training routine  ----------------------------------------
def train_for_batch_size(batch_size, epochs=5, lr=1e-3):
    print(f"\n=== Training with batch size {batch_size} ===")
    model = GRUClassifier(
        len(vocab), 64, 128, len(set(train_dict["label"])), pad_idx
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    record = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    for ep in range(1, epochs + 1):
        model.train()
        tloss = n = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            tloss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
        train_loss = tloss / n
        # validation
        model.eval()
        vloss = n = 0
        all_pred, all_true, all_seq = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = crit(logits, batch["y"])
                vloss += loss.item() * batch["y"].size(0)
                n += batch["y"].size(0)
                preds = logits.argmax(1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw"])
        vloss /= n
        vscwa = scwa(all_seq, all_true, all_pred)
        record["losses"]["train"].append(train_loss)
        record["losses"]["val"].append(vloss)
        record["metrics"]["val"].append(vscwa)
        record["epochs"].append(ep)
        print(
            f"  Epoch {ep}: train_loss={train_loss:.4f} | val_loss={vloss:.4f} | SCWA={vscwa:.4f}"
        )
    # test set evaluation --------------------------------------------------
    model.eval()
    all_pred, all_true, all_seq = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            preds = logits.argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(batch["y"].cpu().tolist())
            all_seq.extend(batch["raw"])
    test_scwa = scwa(all_seq, all_true, all_pred)
    record["predictions"] = all_pred
    record["ground_truth"] = all_true
    record["test_scwa"] = test_scwa
    print(f"Batch size {batch_size} | Test SCWA = {test_scwa:.4f}")
    return record


# -----------------------------------  run sweep  -----------------------------------------------
for bs in [32, 64, 128, 256]:
    rec = train_for_batch_size(bs, epochs=5, lr=1e-3)
    experiment_data["batch_size_tuning"]["SPR_BENCH"][str(bs)] = rec

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll results saved to", os.path.join(working_dir, "experiment_data.npy"))
