import os, pathlib, random, math, time, itertools, sys, warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.cluster import KMeans

# ---------------- basic setup ---------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------- dataset helpers ------------- #
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
    default_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if default_path.exists():
        print("Loading real SPR_BENCH dataset …")
        return load_spr_bench(default_path)

    # ------- fallback synthetic data ---------- #
    print("Real dataset not found – generating synthetic toy data.")
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def gen(n):
        seqs, labels, ids = [], [], []
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


# ---------------- metrics --------------------- #
def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pc_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [w_i if t == p else 0 for w_i, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ---------------- vocab ----------------------- #
def build_vocab(dataset):
    vocab = set()
    for s in dataset["sequence"]:
        vocab.update(s.strip().split())
    return {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0 = PAD


# ---------------- torch dataset --------------- #
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
            torch.nn.functional.pad(
                x["input_ids"], (0, maxlen - len(x["input_ids"])), value=0
            )
            for x in batch
        ]
    )
    labels = torch.stack([x["labels"] for x in batch])
    raw = [x["raw"] for x in batch]
    return {"input_ids": input_ids, "labels": labels, "raw": raw}


# ---------------- model ----------------------- #
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


# ---------------- training loop --------------- #
def run_one_experiment(max_epochs, patience=5):
    # fresh data + vocab each run
    spr = try_load_dataset()
    label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    num_classes = len(label2id)
    stoi = build_vocab(spr["train"])

    # loaders
    def make_loaders(curr_stoi):
        return (
            DataLoader(
                SPRTorchDataset(spr["train"], curr_stoi, label2id),
                batch_size=64,
                shuffle=True,
                collate_fn=collate_f,
            ),
            DataLoader(
                SPRTorchDataset(spr["dev"], curr_stoi, label2id),
                batch_size=128,
                shuffle=False,
                collate_fn=collate_f,
            ),
        )

    train_loader, dev_loader = make_loaders(stoi)

    model = EncoderClassifier(len(stoi) + 1, classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"losses": {"train": [], "val": []}, "metrics": {"train": [], "val": []}}

    best_val, wait, kmeans_done = float("inf"), 0, False

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        total_loss, total_correct, total_items = 0, 0, 0
        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * lab.size(0)
            total_correct += (logits.argmax(1) == lab).sum().item()
            total_items += lab.size(0)
        train_loss = total_loss / total_items
        train_acc = total_correct / total_items
        history["losses"]["train"].append(train_loss)
        history["metrics"]["train"].append({"epoch": epoch, "acc": train_acc})

        # ---- validation ----
        model.eval()
        val_loss, preds, gts, raws = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                inp = batch["input_ids"].to(device)
                lab = batch["labels"].to(device)
                logits = model(inp)
                loss = criterion(logits, lab)
                val_loss += loss.item() * lab.size(0)
                p = logits.argmax(1).cpu().tolist()
                preds.extend(p)
                gts.extend(lab.cpu().tolist())
                raws.extend(batch["raw"])
        val_loss /= len(dev_loader.dataset)
        acc = np.mean([p == l for p, l in zip(preds, gts)])
        cwa = color_weighted_accuracy(raws, gts, preds)
        swa = shape_weighted_accuracy(raws, gts, preds)
        pcwa = pc_weighted_accuracy(raws, gts, preds)
        history["losses"]["val"].append(val_loss)
        history["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": float(acc),
                "cwa": float(cwa),
                "swa": float(swa),
                "pcwa": float(pcwa),
            }
        )
        print(
            f"[{max_epochs}-epoch-budget] Epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} PCWA={pcwa:.3f}"
        )

        # ---------- early stopping -------------
        if val_loss + 1e-4 < best_val:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

        # ---------- latent clustering after 1st epoch ---- #
        if (not kmeans_done) and epoch == 1:
            print("Performing latent glyph clustering …")
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

            # remap sequences
            def remap_ex(ex):
                new_ids = [token_to_cluster[t] for t in ex["sequence"].split()]
                return {"sequence": " ".join(map(str, new_ids))}

            for split in ["train", "dev", "test"]:
                spr[split] = spr[split].map(
                    remap_ex, batched=False, load_from_cache_file=False
                )

            # rebuild stoi and loaders
            stoi = {str(i): i for i in range(1, n_clusters + 1)}
            train_loader, dev_loader = make_loaders(stoi)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            kmeans_done = True
            print(f"Clustering completed. New vocab size: {n_clusters}")

    return history


# ------------- hyper-parameter search --------------- #
experiment_data = {"epoch_tuning": {"SPR_BENCH": {}}}
for max_ep in [20, 25, 30]:
    hist = run_one_experiment(max_ep)
    experiment_data["epoch_tuning"]["SPR_BENCH"][f"max_{max_ep}"] = hist

# ------------- save experiment data ---------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
