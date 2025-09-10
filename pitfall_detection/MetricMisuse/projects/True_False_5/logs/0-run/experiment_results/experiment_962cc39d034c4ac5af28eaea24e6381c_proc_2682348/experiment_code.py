import os, pathlib, time, numpy as np, torch, torch.nn as nn, random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# ---------- work dir / device -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- load original SPR_BENCH ------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if not DATA_PATH.exists():
    DATA_PATH = pathlib.Path("./SPR_BENCH")


def load_csv_split(root, name):  # helper
    return load_dataset(
        "csv",
        data_files=str(root / f"{name}.csv"),
        split="train",
        cache_dir=".cache_dsets",
    )


spr_orig = DatasetDict(
    {k: load_csv_split(DATA_PATH, k) for k in ["train", "dev", "test"]}
)
print("SPR_BENCH sizes:", {k: len(v) for k, v in spr_orig.items()})


# ---------- build synthetic variants -----------------------------------------
def colour_cycle_mapping(colours):
    colours = sorted(colours)
    return {c: colours[(i + 1) % len(colours)] for i, c in enumerate(colours)}


def colour_swap_seq(seq, cmap):
    toks = []
    for tok in seq.strip().split():
        if len(tok) > 1 and tok[1] in cmap:
            toks.append(tok[0] + cmap[tok[1]])
        else:
            toks.append(tok)
    return " ".join(toks)


def size_aug_seq(seq):
    toks = seq.strip().split()
    dup = [t for tok in toks for t in (tok, tok)]  # duplicate each token
    return " ".join(dup)


# collect colour symbols
colour_set = set()
for s in spr_orig["train"]["sequence"]:
    for tok in s.strip().split():
        if len(tok) > 1:
            colour_set.add(tok[1])
cmap = colour_cycle_mapping(colour_set)


def make_variant(base_ds, transform):
    return base_ds.map(lambda ex: {"sequence": transform(ex["sequence"])}, num_proc=1)


spr_colour = DatasetDict(
    {
        k: make_variant(v, lambda s: colour_swap_seq(s, cmap))
        for k, v in spr_orig.items()
    }
)
spr_size = DatasetDict({k: make_variant(v, size_aug_seq) for k, v in spr_orig.items()})
variant_dict = {"SPR_BENCH": spr_orig, "COLOR_SWAP": spr_colour, "SIZE_AUG": spr_size}


# ---------- build global vocab -----------------------------------------------
def build_vocab(seq_iter):
    shapes, colours = set(), set()
    for seq in seq_iter:
        for tok in seq.strip().split():
            shapes.add(tok[0])
            if len(tok) > 1:
                colours.add(tok[1])
    colours.add("<none>")
    shape_map = {"<pad>": 0, **{s: i + 1 for i, s in enumerate(sorted(shapes))}}
    colour_map = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(sorted(colours))}}
    return shape_map, colour_map


all_train_seqs = []
for ds in variant_dict.values():
    all_train_seqs.extend(ds["train"]["sequence"])
shape_map, colour_map = build_vocab(all_train_seqs)
n_shape_sym = len(shape_map) - 1
n_colour_sym = len(colour_map) - 1
sym_dim = n_shape_sym + n_colour_sym
print(f"Vocab: shapes={n_shape_sym}, colours={n_colour_sym}")


# ---------- metrics -----------------------------------------------------------
def count_shape_variety(sequence):  # number of distinct shapes
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


# ---------- torch Dataset -----------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, split, shape_map, colour_map):
        self.seq, self.labels = split["sequence"], split["label"]
        self.shape_map, self.colour_map = shape_map, colour_map
        self.n_shape_sym, self.n_colour_sym = n_shape_sym, n_colour_sym

    def encode_tok(self, tok):
        s = self.shape_map.get(tok[0], 0)
        c = (
            self.colour_map.get(tok[1], self.colour_map["<none>"])
            if len(tok) > 1
            else self.colour_map["<none>"]
        )
        return s, c

    def sym_vec(self, seq):
        s_arr = np.zeros(self.n_shape_sym, dtype=np.float32)
        c_arr = np.zeros(self.n_colour_sym, dtype=np.float32)
        for tok in seq.strip().split():
            if tok[0] != "<pad>":
                s_arr[self.shape_map[tok[0]] - 1] += 1
            if len(tok) > 1 and tok[1] != "<pad>":
                c_arr[self.colour_map[tok[1]] - 1] += 1
        total = max(len(seq.strip().split()), 1)
        return np.concatenate([s_arr, c_arr]) / total

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq_str = self.seq[idx]
        sids, cids = zip(*[self.encode_tok(t) for t in seq_str.strip().split()])
        return dict(
            shape_ids=torch.tensor(sids, dtype=torch.long),
            colour_ids=torch.tensor(cids, dtype=torch.long),
            sym_feats=torch.tensor(self.sym_vec(seq_str), dtype=torch.float32),
            labels=torch.tensor(self.labels[idx], dtype=torch.long),
            sequence_str=seq_str,
        )


def collate(batch):
    shp = nn.utils.rnn.pad_sequence([b["shape_ids"] for b in batch], batch_first=True)
    col = nn.utils.rnn.pad_sequence([b["colour_ids"] for b in batch], batch_first=True)
    mask = shp != 0
    sym = torch.stack([b["sym_feats"] for b in batch])
    lab = torch.stack([b["labels"] for b in batch])
    seqs = [b["sequence_str"] for b in batch]
    return dict(
        shape_ids=shp,
        colour_ids=col,
        attention_mask=mask,
        sym_feats=sym,
        labels=lab,
        sequence_str=seqs,
    )


# ---------- dataloaders -------------------------------------------------------
BATCH = 256
torch_datasets = {
    name: {
        split: SPRDataset(ds[split], shape_map, colour_map)
        for split in ["train", "dev", "test"]
    }
    for name, ds in variant_dict.items()
}
combined_train_ds = ConcatDataset([torch_datasets[n]["train"] for n in torch_datasets])
train_loader = DataLoader(
    combined_train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate
)
val_loaders = {
    n: DataLoader(d["dev"], batch_size=BATCH, shuffle=False, collate_fn=collate)
    for n, d in torch_datasets.items()
}
test_loaders = {
    n: DataLoader(d["test"], batch_size=BATCH, shuffle=False, collate_fn=collate)
    for n, d in torch_datasets.items()
}


# ---------- model -------------------------------------------------------------
class NeuralSymbolicTransformer(nn.Module):
    def __init__(
        self,
        n_shape,
        n_colour,
        sym_dim,
        n_cls,
        d_model=64,
        nhead=4,
        layers=2,
        max_len=128,
    ):
        super().__init__()
        self.shape_emb = nn.Embedding(n_shape, d_model, padding_idx=0)
        self.col_emb = nn.Embedding(n_colour, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + sym_dim, 128), nn.ReLU(), nn.Linear(128, n_cls)
        )

    def forward(self, shape_ids, colour_ids, attn_mask, sym_feats):
        B, T = shape_ids.size()
        pos = torch.arange(T, device=shape_ids.device).unsqueeze(0).expand(B, T)
        tok_emb = (
            self.shape_emb(shape_ids) + self.col_emb(colour_ids) + self.pos_emb(pos)
        )
        enc = self.encoder(tok_emb, src_key_padding_mask=~attn_mask)
        seq_emb = (enc * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(
            1, keepdim=True
        ).clamp(min=1e-5)
        return self.mlp(torch.cat([seq_emb, sym_feats], dim=-1))


num_classes = int(max(torch_datasets["SPR_BENCH"]["train"].labels)) + 1
model = NeuralSymbolicTransformer(
    len(shape_map), len(colour_map), sym_dim, num_classes
).to(device)
criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(
    model.parameters(), lr=1e-3
)


# ---------- evaluate helper ---------------------------------------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, preds, gts, seqs = 0.0, [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(
            batch["shape_ids"],
            batch["colour_ids"],
            batch["attention_mask"],
            batch["sym_feats"],
        )
        loss = criterion(logits, batch["labels"])
        total_loss += loss.item() * batch["labels"].size(0)
        p = logits.argmax(-1).cpu().tolist()
        g = batch["labels"].cpu().tolist()
        preds.extend(p)
        gts.extend(g)
        seqs.extend(batch["sequence_str"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    return total_loss / len(loader.dataset), swa, preds, gts, seqs


# ---------- experiment record -------------------------------------------------
experiment_data = {
    "multi_dataset": {
        n: {
            "metrics": {"train_swa": [], "val_swa": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
        for n in variant_dict
    }
}
exp_rec = experiment_data["multi_dataset"]

# ---------- training loop -----------------------------------------------------
MAX_EPOCHS, patience = 20, 4
best_state, best_avg, no_imp = None, -1.0, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(
            batch["shape_ids"],
            batch["colour_ids"],
            batch["attention_mask"],
            batch["sym_feats"],
        )
        loss = criterion(out, batch["labels"])
        loss.backward()
        optimizer.step()
        running += loss.item() * batch["labels"].size(0)
    train_loss = running / len(train_loader.dataset)

    # evaluate per dataset
    val_sw_total = 0.0
    for name, loader in val_loaders.items():
        train_l, train_swa, *_ = evaluate(
            DataLoader(
                torch_datasets[name]["train"],
                batch_size=BATCH,
                shuffle=False,
                collate_fn=collate,
            )
        )
        val_l, val_swa, *_ = evaluate(loader)
        rec = exp_rec[name]
        rec["losses"]["train"].append(train_l)
        rec["losses"]["val"].append(val_l)
        rec["metrics"]["train_swa"].append(train_swa)
        rec["metrics"]["val_swa"].append(val_swa)
        rec["timestamps"].append(time.time())
        val_sw_total += val_swa
    avg_val_swa = val_sw_total / len(val_loaders)
    print(f"Epoch {epoch:02d}  avg_val_SWA={avg_val_swa:.4f}")

    if avg_val_swa > best_avg:
        best_avg, best_state, no_imp = (
            avg_val_swa,
            {k: v.cpu() for k, v in model.state_dict().items()},
            0,
        )
    else:
        no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break

# ---------- final test --------------------------------------------------------
model.load_state_dict(best_state)
for name, loader in test_loaders.items():
    loss, swa, preds, gts, _ = evaluate(loader)
    rec = exp_rec[name]
    rec["predictions"] = np.array(preds)
    rec["ground_truth"] = np.array(gts)
    print(f"[{name}] TEST  loss={loss:.4f}  SWA={swa:.4f}")

# ---------- save --------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
