import os, pathlib, random, math, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.cluster import KMeans

# ------------------- mandatory dirs & device -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment store -------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------------- SPR load (real or synthetic) ------------------ #
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


def get_dataset() -> DatasetDict:
    real_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if real_path.exists():
        print("Found real SPR_BENCH.")
        return load_spr_bench(real_path)

    # ----------- synthetic fallback ------------
    print("Real benchmark not found, generating synthetic data.")
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def make_split(n):
        ids, seqs, labels = [], [], []
        for i in range(n):
            ids.append(str(i))
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            labels.append(random.choice(["ruleA", "ruleB", "ruleC"]))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labels})

    return DatasetDict(
        train=make_split(2000), dev=make_split(400), test=make_split(400)
    )


spr_raw = get_dataset()


# -------------------- metrics helpers --------------------------- #
def count_color_variety(seq: str) -> int:
    """robustly count different colors (2nd char of glyph)"""
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    """robustly count different shapes (1st char of glyph)"""
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)) / max(
        1, sum(w)
    )


# ------------------ glyph clustering --------------------------- #
def build_token_vectors(vocab):
    shapes = sorted({tok[0] for tok in vocab})
    colors = sorted({tok[1] for tok in vocab})
    shape2i = {s: i for i, s in enumerate(shapes)}
    color2i = {c: i for i, c in enumerate(colors)}
    return np.array([[shape2i[t[0]], color2i[t[1]]] for t in vocab], dtype=float)


def cluster_glyphs(dataset: DatasetDict, k: int = 16):
    vocab = sorted({tok for seq in dataset["train"]["sequence"] for tok in seq.split()})
    vecs = build_token_vectors(vocab)
    k = min(k, len(vocab))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vecs)
    tok2cluster = {tok: str(cid + 1) for tok, cid in zip(vocab, km.labels_)}
    print(f"Clustered {len(vocab)} glyphs into {k} clusters.")

    def remap(example):
        return {
            "cluster_seq": " ".join(tok2cluster[t] for t in example["sequence"].split())
        }

    for split in ["train", "dev", "test"]:
        dataset[split] = dataset[split].map(
            remap, batched=False, load_from_cache_file=False
        )
    return dataset, k


spr_clustered, vocab_size = cluster_glyphs(spr_raw, k=16)


# -------------------- torch dataset ----------------------------- #
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, vocab_sz, label2id):
        self.input_seqs = hf_split["cluster_seq"]
        self.raw_seqs = hf_split["sequence"]  # original glyphs for metrics
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.vocab_sz = vocab_sz

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = [int(tok) for tok in self.input_seqs[idx].split()]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.raw_seqs[idx],
        }


def collate(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    pad = lambda t: nn.functional.pad(t, (0, maxlen - len(t)), value=0)
    input_ids = torch.stack([pad(x["input_ids"]) for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    raw = [x["raw"] for x in batch]
    return {"input_ids": input_ids, "labels": labels, "raw": raw}


# ------------------------ model --------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_sz: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        n_cls: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz + 1, d_model, padding_idx=0)
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, n_cls)

    def forward(self, ids):
        mask = ids == 0
        x = self.embedding(ids)
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0).mean(dim=1)
        return self.fc(x)


# ------------------------ training ------------------------------ #
def run(lr=1e-3, epochs=6):
    print(f"\n=== LR {lr} ===")
    label2id = {
        l: i for i, l in enumerate(sorted(set(spr_clustered["train"]["label"])))
    }
    id2label = {i: l for l, i in label2id.items()}

    train_loader = DataLoader(
        SPRTorchDataset(spr_clustered["train"], vocab_size, label2id),
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr_clustered["dev"], vocab_size, label2id),
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
    )

    model = TransformerClassifier(vocab_size, n_cls=len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch["labels"].size(0)
        train_loss = total_loss / len(train_loader.dataset)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)

        # ---- validation ----
        model.eval()
        val_loss, preds, gts, raws = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                raws.extend(batch["raw"])
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item() * batch["labels"].size(0)
                p = torch.argmax(logits, 1).cpu().tolist()
                preds.extend(p)
                gts.extend(batch["labels"].cpu().tolist())
        val_loss /= len(dev_loader.dataset)
        acc = np.mean([p == g for p, g in zip(preds, gts)])
        cwa = color_weighted_accuracy(raws, gts, preds)
        swa = shape_weighted_accuracy(raws, gts, preds)
        comp = complexity_weighted_accuracy(raws, gts, preds)

        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": float(acc),
                "cwa": float(cwa),
                "swa": float(swa),
                "compwa": float(comp),
            }
        )

        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} | "
            f"ACC {acc:.3f} CWA {cwa:.3f} SWA {swa:.3f} CompWA {comp:.3f}"
        )

    # store final predictions
    experiment_data["SPR_BENCH"]["predictions"] = preds
    experiment_data["SPR_BENCH"]["ground_truth"] = gts


for lr in [5e-4, 1e-3, 2e-3]:
    run(lr=lr, epochs=6)

# -------------------- save results ------------------------------ #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
