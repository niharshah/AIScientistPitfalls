# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, random, numpy as np, torch, copy
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- working dir & GPU ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment store ----------
experiment_data = {}


# ---------- load SPR_BENCH or synthetic fallback ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def get_dataset() -> DatasetDict:
    spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if spr_path.exists():
        print("Loading real SPR_BENCH …")
        return load_spr_bench(spr_path)
    print("Real SPR_BENCH not found — generating toy data.")
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def gen(n):
        seqs, labels, ids = [], [], []
        for i in range(n):
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            labels.append(random.choice(["ruleA", "ruleB", "ruleC"]))
            ids.append(str(i))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labels})

    return DatasetDict(train=gen(800), dev=gen(200), test=gen(200))


spr = get_dataset()


# ---------- metrics ----------
def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def SWA(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def CompWA(seqs, y_t, y_p):
    w = [count_color(s) * count_shape(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------- vocabulary ----------
def make_stoi(split):
    vocab = set()
    for s in split["sequence"]:
        vocab.update(s.split())
    return {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0 = PAD


# ---------- torch dataset ----------
class SPRTorch(torch.utils.data.Dataset):
    def __init__(self, hf_split, stoi, label2id):
        self.raw_seqs = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]
        self.stoi = stoi

    def __len__(self):
        return len(self.raw_seqs)

    def __getitem__(self, idx):
        ids = [self.stoi[tok] for tok in self.raw_seqs[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(self.labels[idx]),
            "raw": self.raw_seqs[idx],
        }


def collate(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    inp = torch.stack(
        [
            nn.functional.pad(
                x["input_ids"], (0, maxlen - len(x["input_ids"])), value=0
            )
            for x in batch
        ]
    )
    lbl = torch.stack([x["labels"] for x in batch])
    raw = [x["raw"] for x in batch]
    return {"input_ids": inp, "labels": lbl, "raw": raw}


# ---------- model ----------
class EncoderClassifier(nn.Module):
    def __init__(self, vocab, emb=32, hidden=64, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        e = self.embedding(x)
        _, h = self.rnn(e)
        h = torch.cat([h[0], h[1]], 1)
        return self.fc(h)


# ---------- training ----------
def train_loop(lr=2e-3, epochs=4):
    stoi = make_stoi(spr["train"])
    label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    train_dl = DataLoader(
        SPRTorch(spr["train"], stoi, label2id),
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
    dev_dl = DataLoader(
        SPRTorch(spr["dev"], stoi, label2id),
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
    )
    model = EncoderClassifier(len(stoi) + 1, classes=len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    exp = "baseline_cluster"
    experiment_data[exp] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    clustered = False
    for epoch in range(1, epochs + 1):
        # --- train
        model.train()
        tot = 0
        for batch in train_dl:
            batch_t = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            opt.zero_grad()
            out = model(batch_t["input_ids"])
            loss = loss_fn(out, batch_t["labels"])
            loss.backward()
            opt.step()
            tot += loss.item() * batch_t["labels"].size(0)
        tr_loss = tot / len(train_dl.dataset)
        experiment_data[exp]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # --- validate
        model.eval()
        vloss = 0
        preds = []
        gts = []
        raws = []
        with torch.no_grad():
            for batch in dev_dl:
                batch_t = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                out = model(batch_t["input_ids"])
                loss = loss_fn(out, batch_t["labels"])
                vloss += loss.item() * batch_t["labels"].size(0)
                p = torch.argmax(out, 1).cpu().tolist()
                preds.extend(p)
                gts.extend(batch_t["labels"].cpu().tolist())
                raws.extend(batch["raw"])
        vloss /= len(dev_dl.dataset)
        acc = float(np.mean([p == g for p, g in zip(preds, gts)]))
        cwa = float(CWA(raws, gts, preds))
        swa = float(SWA(raws, gts, preds))
        comp = float(CompWA(raws, gts, preds))
        experiment_data[exp]["SPR_BENCH"]["losses"]["val"].append(vloss)
        experiment_data[exp]["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "CWA": cwa, "SWA": swa, "CompWA": comp}
        )
        print(
            f"Epoch {epoch}: validation_loss = {vloss:.4f} | ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )

        # --- run clustering once after epoch 1 ---
        if not clustered:
            print("Running latent glyph clustering …")
            with torch.no_grad():
                emb_np = model.embedding.weight[1:].detach().cpu().numpy()
            k = min(16, emb_np.shape[0])
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb_np)
            token_to_cluster = {
                tok: cid + 1
                for tok, cid in zip(sorted(stoi, key=lambda x: stoi[x]), km.labels_)
            }
            new_emb = nn.Embedding(
                k + 1, model.embedding.embedding_dim, padding_idx=0
            ).to(device)
            new_emb.weight.data[1:] = torch.tensor(km.cluster_centers_, device=device)
            model.embedding = new_emb  # replace
            # rebuild stoi
            stoi = {tok: token_to_cluster[tok] for tok in stoi}
            train_dl = DataLoader(
                SPRTorch(spr["train"], stoi, label2id),
                batch_size=64,
                shuffle=True,
                collate_fn=collate,
            )
            dev_dl = DataLoader(
                SPRTorch(spr["dev"], stoi, label2id),
                batch_size=128,
                shuffle=False,
                collate_fn=collate,
            )
            clustered = True
            print(f"Clustering finished. New vocab (clusters) = {k}")
    # store final preds / gts
    experiment_data[exp]["SPR_BENCH"]["predictions"] = preds
    experiment_data[exp]["SPR_BENCH"]["ground_truth"] = gts


train_loop()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
