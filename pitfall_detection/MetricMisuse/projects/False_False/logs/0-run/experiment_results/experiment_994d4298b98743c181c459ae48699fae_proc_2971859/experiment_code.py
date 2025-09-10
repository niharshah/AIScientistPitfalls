import os, time, random, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from collections import Counter

# -------------------- I/O & ENV --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- DATA -------------------------
def resolve_spr_path():
    for p in [
        os.getenv("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.cwd().parent / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if p and (pathlib.Path(p) / "train.csv").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("Could not locate SPR_BENCH")


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


spr = load_spr_bench(resolve_spr_path())
print({k: len(v) for k, v in spr.items()})


# -------------------- VOCAB ------------------------
def tokenize(s):
    return s.strip().split()


vocab_counter = Counter(tok for s in spr["train"]["sequence"] for tok in tokenize(s))
vocab = ["<PAD>", "<UNK>"] + sorted(vocab_counter)
stoi = {w: i for i, w in enumerate(vocab)}
pad_idx, unk_idx = 0, 1


def encode_tokens(toks):
    return [stoi.get(t, unk_idx) for t in toks]


def encode_seq(seq):
    return encode_tokens(tokenize(seq))


labels = sorted(set(spr["train"]["label"]))
ltoi, itos_l = {l: i for i, l in enumerate(labels)}, {
    i: l for l, i in enumerate(labels)
}


# -------------------- METRICS ----------------------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def weighted_acc(weights, y_true, y_pred):
    w = sum(weights)
    c = sum(wi for wi, t, p in zip(weights, y_true, y_pred) if t == p)
    return c / w if w else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc([count_shape_variety(s) for s in seqs], y_true, y_pred)


def color_weighted_accuracy(seqs, y_true, y_pred):
    return weighted_acc([count_color_variety(s) for s in seqs], y_true, y_pred)


def scwa(seqs, y_true, y_pred):
    return weighted_acc(
        [count_shape_variety(s) + count_color_variety(s) for s in seqs], y_true, y_pred
    )


# -------------------- AUGMENTATION -----------------
def augment_tokens(toks):
    toks = [t for t in toks if random.random() > 0.15] or toks
    if len(toks) > 3 and random.random() < 0.3:
        i, j = sorted(random.sample(range(len(toks)), 2))
        toks[i:j] = reversed(toks[i:j])
    return toks


# -------------------- DATASETS ---------------------
MAX_LEN = 128


class ContrastiveSPR(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_contrastive(batch):
    views = []
    for s in batch:
        tok = tokenize(s)
        views.append(encode_tokens(augment_tokens(tok)))
        views.append(encode_tokens(augment_tokens(tok)))
    maxlen = min(MAX_LEN, max(len(v) for v in views))
    x = torch.full((len(views), maxlen), pad_idx, dtype=torch.long)
    for i, seq in enumerate(views):
        x[i, : len(seq[:maxlen])] = torch.tensor(seq[:maxlen])
    return x.to(device)


class SupervisedSPR(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labs = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(
                encode_seq(self.seqs[idx])[:MAX_LEN], dtype=torch.long
            ),
            "label": torch.tensor(self.labs[idx], dtype=torch.long),
        }


def collate_supervised(batch):
    maxlen = max(len(b["input"]) for b in batch)
    x = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["input"])] = b["input"]
    y = torch.stack([b["label"] for b in batch])
    return {"input": x.to(device), "label": y.to(device)}


# -------------------- MODEL ------------------------
def sinusoid_table(n_pos, d_hid):
    def angle(i, pos):
        return pos / np.power(10000, 2 * (i // 2) / d_hid)

    table = np.array(
        [[angle(i, pos) for i in range(d_hid)] for pos in range(n_pos)],
        dtype=np.float32,
    )
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.from_numpy(table)


class SPRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        n_heads=4,
        n_layers=2,
        max_len=MAX_LEN,
        pos_type="learned",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_type = pos_type
        if pos_type == "learned":
            self.pos = nn.Embedding(max_len, emb_dim)
        else:
            self.register_buffer("pos_table", sinusoid_table(max_len, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.emb_dim = emb_dim

    def forward(self, x):
        pos_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + (
            self.pos(pos_ids) if self.pos_type == "learned" else self.pos_table[pos_ids]
        )
        mask = x == pad_idx
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1)
        pooled = (h * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        return pooled


class ProjectionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, z):
        return self.fc(z)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.emb_dim, num_labels)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def nt_xent(z, temp=0.5):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-9e15)
    N = z.size(0) // 2
    pos = torch.arange(sim.size(0), device=z.device)
    pos = torch.where(pos < N, pos + N, pos - N)
    return F.cross_entropy(sim, pos)


# -------------------- TRAINING LOOP ---------------
def run_experiment(ablation_key, pos_type, pre_epochs=2, fine_epochs=4):
    data = {
        "metrics": {"val_SWA": [], "val_CWA": [], "val_SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    encoder = SPRTransformer(len(vocab), emb_dim=128, pos_type=pos_type).to(device)
    proj = ProjectionHead(128).to(device)
    opt_pre = torch.optim.Adam(
        list(encoder.parameters()) + list(proj.parameters()), lr=1e-3
    )
    pre_loader = DataLoader(
        ContrastiveSPR(spr["train"]["sequence"]),
        batch_size=256,
        shuffle=True,
        collate_fn=collate_contrastive,
    )
    for ep in range(1, pre_epochs + 1):
        encoder.train()
        proj.train()
        running = 0.0
        for xb in pre_loader:
            opt_pre.zero_grad()
            loss = nt_xent(proj(encoder(xb)))
            loss.backward()
            opt_pre.step()
            running += loss.item() * xb.size(0)
        data["losses"]["pretrain"].append(running / len(pre_loader.dataset))
        print(
            f"[{ablation_key}] Pretrain Epoch {ep}: loss={data['losses']['pretrain'][-1]:.4f}"
        )
    model = SPRModel(encoder, len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        SupervisedSPR(spr["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_supervised,
    )
    val_loader = DataLoader(
        SupervisedSPR(spr["dev"]),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_supervised,
    )
    best_scwa = -1
    best_preds = []
    best_trues = []
    for ep in range(1, fine_epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            loss = crit(model(batch["input"]), batch["label"])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch["label"].size(0)
        tr_loss /= len(train_loader.dataset)
        data["losses"]["train"].append(tr_loss)
        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input"])
                val_loss += crit(logits, batch["label"]).item() * batch["label"].size(0)
                preds += logits.argmax(1).cpu().tolist()
                trues += batch["label"].cpu().tolist()
        val_loss /= len(val_loader.dataset)
        data["losses"]["val"].append(val_loss)
        swa = shape_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
        cwa = color_weighted_accuracy(spr["dev"]["sequence"], trues, preds)
        sc = scwa(spr["dev"]["sequence"], trues, preds)
        data["metrics"]["val_SWA"].append(swa)
        data["metrics"]["val_CWA"].append(cwa)
        data["metrics"]["val_SCWA"].append(sc)
        data["timestamps"].append(time.time())
        if sc > best_scwa:
            best_scwa = sc
            best_preds, preds
            best_trues, trues
        print(
            f"[{ablation_key}] Epoch {ep}: val_loss={val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} SCWA={sc:.4f}"
        )
    data["predictions"] = best_preds
    data["ground_truth"] = best_trues
    return data


# -------------------- RUN BOTH VARIANTS ------------
experiment_data = {
    "learned_positional_embeddings": {"SPR_BENCH": {}},
    "sinusoidal_positional_embeddings": {"SPR_BENCH": {}},
}
experiment_data["learned_positional_embeddings"]["SPR_BENCH"] = run_experiment(
    "learned_positional_embeddings", "learned"
)
experiment_data["sinusoidal_positional_embeddings"]["SPR_BENCH"] = run_experiment(
    "sinusoidal_positional_embeddings", "sinusoidal"
)

# -------------------- SAVE -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved to working/experiment_data.npy")
