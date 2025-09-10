import os, random, json, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.cluster import KMeans

# --------------------------------------------- misc & device ------------------------------------------------ #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# -------------------------------------- metric helpers (unchanged) ------------------------------------------ #
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def glyph_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------------------------------------- synthetic datasets ------------------------------------------------ #
#  three generators with different inventories, length distributions, and label rules
cfgs = {
    "DataA": {
        "shapes": list("ABCD"),
        "colors": list("1234"),
        "len_rng": (3, 9),
        "rule": "random",
    },
    "DataB": {
        "shapes": list("EFGHIJ"),
        "colors": list("567890"),
        "len_rng": (5, 15),
        "rule": "shape_sum",
    },
    "DataC": {
        "shapes": list("KLM"),
        "colors": list("abc"),
        "len_rng": (2, 6),
        "rule": "color_sum",
    },
}
num_classes = 4


def _label(seq, cfg):
    if cfg["rule"] == "random":
        return random.randint(0, num_classes - 1)
    elif cfg["rule"] == "shape_sum":
        return sum(cfg["shapes"].index(tok[0]) for tok in seq.split()) % num_classes
    elif cfg["rule"] == "color_sum":
        return sum(cfg["colors"].index(tok[1]) for tok in seq.split()) % num_classes
    raise ValueError


def gen_split(cfg, n, start_id):
    rows = []
    for i in range(n):
        ln = random.randint(*cfg["len_rng"])
        seq = " ".join(
            random.choice(cfg["shapes"]) + random.choice(cfg["colors"])
            for _ in range(ln)
        )
        rows.append({"id": start_id + i, "sequence": seq, "label": _label(seq, cfg)})
    return rows


datasets = {}
id_counter = 0
for name, cfg in cfgs.items():
    splits = {}
    for sp, n in [("train", 800), ("dev", 200), ("test", 200)]:
        splits[sp] = gen_split(cfg, n, id_counter)
        id_counter += n
    datasets[name] = splits

# ------------------------------------- global glyph statistics --------------------------------------------- #
all_tokens = [
    tok for d in datasets.values() for s in d["train"] for tok in s["sequence"].split()
]
all_shapes = sorted({t[0] for t in all_tokens})
all_colors = sorted({t[1] for t in all_tokens})
shape2id = {s: i + 1 for i, s in enumerate(all_shapes)}
color2id = {c: i + 1 for i, c in enumerate(all_colors)}

token_set = sorted(set(all_tokens))
token_vecs = np.array(
    [[shape2id[t[0]], color2id[t[1]]] for t in token_set], dtype=float
)
n_clusters = min(max(6, len(token_vecs) // 2), 40)
print(f"Clustering {len(token_vecs)} glyphs into {n_clusters} clusters")
tok2cluster = {
    tok: int(c) + 1
    for tok, c in zip(
        token_set, KMeans(n_clusters=n_clusters, random_state=0).fit(token_vecs).labels_
    )
}


# ------------------------------------------- torch Datasets ------------------------------------------------- #
class SynthDataset(Dataset):
    def __init__(self, records):
        self.seq = [r["sequence"] for r in records]
        self.lab = [r["label"] for r in records]

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        tokens = self.seq[idx].split()
        return {
            "shape": [shape2id[t[0]] for t in tokens],
            "color": [color2id[t[1]] for t in tokens],
            "cluster": [tok2cluster[t] for t in tokens],
            "label": self.lab[idx],
            "seq_str": self.seq[idx],
        }


def collate(batch):
    maxlen = max(len(b["shape"]) for b in batch)

    def pad(key):
        return torch.tensor(
            [b[key] + [0] * (maxlen - len(b[key])) for b in batch], dtype=torch.long
        )

    out = {
        "shape": pad("shape"),
        "color": pad("color"),
        "cluster": pad("cluster"),
        "mask": (pad("shape") != 0).float(),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "seqs": [b["seq_str"] for b in batch],
    }
    return out


batch_size = 128
train_concat = ConcatDataset([SynthDataset(datasets[name]["train"]) for name in cfgs])
train_loader = DataLoader(
    train_concat, batch_size=batch_size, shuffle=True, collate_fn=collate
)

dev_loaders = {
    name: DataLoader(
        SynthDataset(datasets[name]["dev"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    for name in cfgs
}
test_loaders = {
    name: DataLoader(
        SynthDataset(datasets[name]["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    for name in cfgs
}


# ------------------------------------------------ model ------------------------------------------------------ #
class BiLSTMClassifier(nn.Module):
    def __init__(
        self, n_shape, n_color, n_cluster, num_classes, emb_dim=32, hidden=64, dropp=0.2
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape + 1, emb_dim, padding_idx=0)
        self.color_emb = nn.Embedding(n_color + 1, emb_dim, padding_idx=0)
        self.clus_emb = nn.Embedding(n_cluster + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim * 3,
            hidden_size=hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropp)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropp),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, sh, co, cl, mask):
        x = torch.cat(
            [self.shape_emb(sh), self.color_emb(co), self.clus_emb(cl)], dim=-1
        )
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        masked = out * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1, keepdim=True)
        return self.fc(self.dropout(pooled))


model = BiLSTMClassifier(len(all_shapes), len(all_colors), n_clusters, num_classes).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10


# ----------------------------- evaluation helper ------------------------------------------------------------- #
def evaluate(net, loader):
    net.eval()
    all_preds, all_tgts, all_seqs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = net(bt["shape"], bt["color"], bt["cluster"], bt["mask"])
            loss = criterion(logits, bt["labels"])
            total_loss += loss.item() * bt["labels"].size(0)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(bt["labels"].cpu().tolist())
            all_seqs.extend(batch["seqs"])
    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "CWA": color_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "SWA": shape_weighted_accuracy(all_seqs, all_tgts, all_preds),
        "GCWA": glyph_complexity_weighted_accuracy(all_seqs, all_tgts, all_preds),
    }
    return avg_loss, metrics, all_preds, all_tgts


# --------------------------------------- storage dict -------------------------------------------------------- #
experiment_data = {"multi_synth": {}}
for name in cfgs:
    experiment_data["multi_synth"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# --------------------------------------- training loop ------------------------------------------------------- #
for epoch in range(1, epochs + 1):
    model.train()
    run_loss = 0.0
    for batch in train_loader:
        bt = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(bt["shape"], bt["color"], bt["cluster"], bt["mask"])
        loss = criterion(logits, bt["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss += loss.item() * bt["labels"].size(0)
    train_loss = run_loss / len(train_loader.dataset)

    # evaluate on each dev split
    for name, loader in dev_loaders.items():
        val_loss, val_metrics, _, _ = evaluate(model, loader)
        experiment_data["multi_synth"][name]["losses"]["train"].append(train_loss)
        experiment_data["multi_synth"][name]["losses"]["val"].append(val_loss)
        experiment_data["multi_synth"][name]["metrics"]["train"].append({})
        experiment_data["multi_synth"][name]["metrics"]["val"].append(val_metrics)
    print(f"[Epoch {epoch}] TrainLoss={train_loss:.4f}")
    for name in cfgs:
        m = experiment_data["multi_synth"][name]["metrics"]["val"][-1]
        print(
            f"  {name}: val_loss={experiment_data['multi_synth'][name]['losses']['val'][-1]:.4f} | CWA={m['CWA']:.3f} | SWA={m['SWA']:.3f} | GCWA={m['GCWA']:.3f}"
        )

# ----------------------------------------- final test -------------------------------------------------------- #
for name, loader in test_loaders.items():
    tst_loss, tst_metrics, preds, tgts = evaluate(model, loader)
    experiment_data["multi_synth"][name]["losses"]["test"] = tst_loss
    experiment_data["multi_synth"][name]["metrics"]["test"] = tst_metrics
    experiment_data["multi_synth"][name]["predictions"] = preds
    experiment_data["multi_synth"][name]["ground_truth"] = tgts
    print(
        f"[TEST] {name}: loss={tst_loss:.4f} | CWA={tst_metrics['CWA']:.3f} | SWA={tst_metrics['SWA']:.3f} | GCWA={tst_metrics['GCWA']:.3f}"
    )

# -------------------------------------------- save ----------------------------------------------------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
