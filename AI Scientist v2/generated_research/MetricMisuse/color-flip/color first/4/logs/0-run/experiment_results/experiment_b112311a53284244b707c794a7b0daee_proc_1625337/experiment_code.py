import os, pathlib, random, math, time, itertools, sys, warnings
import numpy as np, torch, sklearn
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# -------------------- data helpers -------------------------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def try_load_dataset() -> DatasetDict:
    default = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if default.exists():
        print("Loading real SPR_BENCH …")
        return load_spr_bench(default)
    # ----------- synthetic fallback -----------
    print("Real dataset not found – creating synthetic data.")
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def gen(n):
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

    return DatasetDict(train=gen(500), dev=gen(100), test=gen(100))


def build_vocab(dataset):
    vocab = set()
    for s in dataset["sequence"]:
        vocab.update(s.strip().split())
    return {tok: i + 1 for i, tok in enumerate(sorted(vocab))}


# ------------------- metrics -------------------------------------- #
def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split() if t))


def _weighted_acc(seqs, y_t, y_p, weight_fn):
    w = [weight_fn(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_color_variety)


def shape_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(s, y_t, y_p, count_shape_variety)


def pc_weighted_accuracy(s, y_t, y_p):
    return _weighted_acc(
        s, y_t, y_p, lambda x: count_color_variety(x) + count_shape_variety(x)
    )


# ------------------ torch dataset -------------------------------- #
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, stoi_dict, label2id):
        self.seqs = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.stoi = stoi_dict

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [self.stoi[t] for t in self.seqs[idx].split()]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.seqs[idx],
        }


def collate_f(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.stack(
        [
            nn.functional.pad(
                x["input_ids"], (0, maxlen - len(x["input_ids"])), value=0
            )
            for x in batch
        ]
    )
    labels = torch.stack([x["labels"] for x in batch])
    raw = [x["raw"] for x in batch]
    return {"input_ids": input_ids, "labels": labels, "raw": raw}


# ------------------ model ---------------------------------------- #
class EncoderClassifier(nn.Module):
    def __init__(self, vocab, embed_dim=32, hidden=64, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.lin(h)


# ------------------ experiment container ------------------------- #
experiment_data = {"batch_size_tuning": {}}


# ------------------ training run per batch size ------------------ #
def run_experiment(train_bs):
    print(f"\n=== Running experiment with TRAIN batch_size={train_bs} ===")
    spr = try_load_dataset()  # fresh dataset
    stoi = build_vocab(spr["train"])
    itos = {i: t for t, i in stoi.items()}
    label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    num_classes = len(label2id)

    train_loader = DataLoader(
        SPRTorchDataset(spr["train"], stoi, label2id),
        batch_size=train_bs,
        shuffle=True,
        collate_fn=collate_f,
    )
    dev_loader = DataLoader(
        SPRTorchDataset(spr["dev"], stoi, label2id),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_f,
    )

    model = EncoderClassifier(len(stoi) + 1, classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 5
    kmeans_done = False
    store = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ------ train -------
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch["labels"].size(0)
        train_loss = total_loss / len(train_loader.dataset)
        store["losses"]["train"].append(train_loss)

        # ------ val ---------
        model.eval()
        val_loss = 0
        preds = []
        labels = []
        raws = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"])
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item() * batch["labels"].size(0)
                p = torch.argmax(logits, 1).cpu().tolist()
                preds.extend(p)
                labels.extend(batch["labels"].cpu().tolist())
                raws.extend(batch["raw"])
        val_loss /= len(dev_loader.dataset)
        store["losses"]["val"].append(val_loss)
        acc = float(np.mean([p == l for p, l in zip(preds, labels)]))
        cwa = float(color_weighted_accuracy(raws, labels, preds))
        swa = float(shape_weighted_accuracy(raws, labels, preds))
        pcwa = float(pc_weighted_accuracy(raws, labels, preds))
        store["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "cwa": cwa, "swa": swa, "pcwa": pcwa}
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} PCWA={pcwa:.3f}"
        )

        # -------- latent clustering after first epoch -------------
        if (not kmeans_done) and epoch == 1:
            print("Latent glyph clustering …")
            with torch.no_grad():
                emb_np = model.embedding.weight.detach().cpu().numpy()[1:]
            n_clusters = min(16, emb_np.shape[0])
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(emb_np)
            token_to_cluster = {
                tok: int(cid) + 1
                for tok, cid in zip(sorted(stoi, key=lambda t: stoi[t]), km.labels_)
            }
            new_emb = nn.Embedding(
                n_clusters + 1, model.embedding.embedding_dim, padding_idx=0
            ).to(device)
            new_emb.weight.data[1:] = torch.tensor(
                km.cluster_centers_, dtype=torch.float32, device=device
            )
            model.embedding = new_emb

            def remap_ex(ex):
                new_ids = [token_to_cluster[t] for t in ex["sequence"].split()]
                return {"sequence": " ".join(map(str, new_ids))}

            for split in ["train", "dev", "test"]:
                spr[split] = spr[split].map(
                    remap_ex, batched=False, load_from_cache_file=False
                )

            stoi = {str(i): i for i in range(1, n_clusters + 1)}
            train_loader = DataLoader(
                SPRTorchDataset(spr["train"], stoi, label2id),
                batch_size=train_bs,
                shuffle=True,
                collate_fn=collate_f,
            )
            dev_loader = DataLoader(
                SPRTorchDataset(spr["dev"], stoi, label2id),
                batch_size=128,
                shuffle=False,
                collate_fn=collate_f,
            )
            kmeans_done = True
            print("Clustering completed. New vocab size:", n_clusters)
    # store final preds/labels
    store["predictions"] = preds
    store["ground_truth"] = labels
    return store


# ------------------- run tuning ---------------------------------- #
for bs in [32, 64, 128]:
    experiment_data["batch_size_tuning"][f"bs_{bs}"] = run_experiment(bs)

# ------------------- save ---------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
