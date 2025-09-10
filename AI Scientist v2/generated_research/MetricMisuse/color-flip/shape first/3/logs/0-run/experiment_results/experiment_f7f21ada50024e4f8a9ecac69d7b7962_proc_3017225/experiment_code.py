import os, random, pathlib, numpy as np, torch, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------- paths / dirs -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------- device ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------- util functions ----------------------------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def scwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    return sum(w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)) / (
        sum(weights) + 1e-9
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
            _sp.split(".")[0]: _ld(_sp) for _sp in ["train.csv", "dev.csv", "test.csv"]
        }
    except Exception as e:
        print("Falling back to synthetic data:", e)
        return False, {}


# -------------------------------- synthetic data ----------------------------------------------
def make_synth_dataset(n_rows):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n_rows):
        L = random.randint(3, 10)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) > count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


# -------------------------------- dataset wrapper ---------------------------------------------
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
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]
        ids = ids[: self.max_len] + [self.vocab["<pad>"]] * max(
            0, self.max_len - len(ids)
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "x": self.encode(self.seqs[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


# -------------------------------- model -------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_classes, pad_idx, dropout_rate):
        super().__init__()
        self.do = nn.Dropout(dropout_rate)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        emb = self.do(self.emb(x))  # dropout on embeddings
        _, h = self.gru(emb)
        h = self.do(h.squeeze(0))  # dropout on hidden
        return self.fc(h)


# -------------------------------- data prep ---------------------------------------------------
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

all_tokens = sorted({tok for seq in train_dict["sequence"] for tok in seq.split()})
vocab = {tok: i + 2 for i, tok in enumerate(all_tokens)}
vocab["<pad>"], vocab["<unk>"] = 0, 1
pad_idx, max_len = vocab["<pad>"], max(len(s.split()) for s in train_dict["sequence"])

train_ds = SPRDataset(train_dict["sequence"], train_dict["label"], vocab, max_len)
dev_ds = SPRDataset(dev_dict["sequence"], dev_dict["label"], vocab, max_len)
test_ds = SPRDataset(test_dict["sequence"], test_dict["label"], vocab, max_len)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

# -------------------------------- experiment bookkeeping --------------------------------------
experiment_data = {"dropout_tuning": {}}

# -------------------------------- hyperparameter sweep ----------------------------------------
dropout_rates = [0.0, 0.2, 0.4, 0.6]
EPOCHS = 5
for rate in dropout_rates:
    tag = f"rate_{rate}"
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "dropout_rate": rate,
    }
    print(f"\n=== training with dropout={rate} ===")
    model = GRUClassifier(
        len(vocab), 64, 128, len(set(train_dict["label"])), pad_idx, rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tot_loss = n = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss = criterion(model(batch["x"]), batch["y"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
        train_loss = tot_loss / n
        exp["losses"]["train"].append(train_loss)

        # ---- val ----
        model.eval()
        val_loss = n = 0
        all_p, all_t, all_s = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = criterion(logits, batch["y"])
                val_loss += loss.item() * batch["y"].size(0)
                n += batch["y"].size(0)
                pred = logits.argmax(1).cpu().tolist()
                all_p.extend(pred)
                all_t.extend(batch["y"].cpu().tolist())
                all_s.extend(batch["raw"])
        val_loss /= n
        val_scwa = scwa(all_s, all_t, all_p)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["val"].append(val_scwa)
        exp["epochs"].append(epoch)
        print(
            f"  epoch {epoch}: train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_SCWA {val_scwa:.4f}"
        )

    # ---- final test ----
    model.eval()
    all_p, all_t, all_s = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["x"])
            all_p.extend(logits.argmax(1).cpu().tolist())
            all_t.extend(batch["y"].cpu().tolist())
            all_s.extend(batch["raw"])
    test_scwa = scwa(all_s, all_t, all_p)
    exp["metrics"]["test_SCWA"] = test_scwa
    exp["predictions"], exp["ground_truth"] = all_p, all_t
    print(f"  --> test_SCWA {test_scwa:.4f}")

    experiment_data["dropout_tuning"][tag] = exp

# -------------------------------- save results -------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished, results saved to working/experiment_data.npy")
