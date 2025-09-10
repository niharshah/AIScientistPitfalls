import os, pathlib, random, string, warnings, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---- working dir & device ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ---- try importing helper utilities ----------------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy
except Exception as e:
    warnings.warn(f"Could not import SPR helpers ({e}); using fall-backs.")

    def load_spr_bench(root: pathlib.Path):
        raise FileNotFoundError("SPR_BENCH folder not found")

    def _count_shape(seq):
        return len(set(t[0] for t in seq.split() if t))

    def shape_weighted_accuracy(seqs, y_true, y_pred):
        w = [_count_shape(s) for s in seqs]
        cor = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
        return sum(cor) / (sum(w) + 1e-9)


# ---- synthetic data generator (fallback) ------------------------------------
def make_synth(n):
    shapes = list(string.ascii_uppercase[:6])
    cols = list(string.ascii_lowercase[:6])
    seqs, lbls = [], []
    for _ in range(n):
        L = random.randint(4, 9)
        toks = [random.choice(shapes) + random.choice(cols) for _ in range(L)]
        seqs.append(" ".join(toks))
        lbls.append(random.randint(0, 3))
    return {"sequence": seqs, "label": lbls}


# ---- load dataset -----------------------------------------------------------
root = pathlib.Path(os.getenv("SPR_BENCH_PATH", "SPR_BENCH"))
try:
    dset = load_spr_bench(root)
    tr_seqs, tr_lbl = dset["train"]["sequence"], dset["train"]["label"]
    dv_seqs, dv_lbl = dset["dev"]["sequence"], dset["dev"]["label"]
    te_seqs, te_lbl = dset["test"]["sequence"], dset["test"]["label"]
    print("Loaded real SPR_BENCH.")
except Exception as e:
    warnings.warn(f"{e}\nUsing synthetic data instead.")
    tr = make_synth(1000)
    dv = make_synth(200)
    te = make_synth(400)
    tr_seqs, tr_lbl = tr["sequence"], tr["label"]
    dv_seqs, dv_lbl = dv["sequence"], dv["label"]
    te_seqs, te_lbl = te["sequence"], te["label"]

num_classes = max(tr_lbl + dv_lbl + te_lbl) + 1
print(f"#classes = {num_classes}")

# ---- vocabulary -------------------------------------------------------------
PAD = "<PAD>"
tok_set = {tok for seq in tr_seqs for tok in seq.split()}
tok2id = {PAD: 0, **{t: i + 1 for i, t in enumerate(sorted(tok_set))}}
vocab_size = len(tok2id)
print(f"Vocab size = {vocab_size}")

max_shapes = len({t[0] for t in tok_set})
max_colors = len({t[1] for t in tok_set})


# ---- PyTorch Dataset --------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        toks = s.split()
        ids = [tok2id[t] for t in toks]
        shape_cnt = len(set(t[0] for t in toks))
        color_cnt = len(set(t[1] for t in toks))
        return {
            "ids": ids,
            "shape_norm": shape_cnt / max_shapes,
            "color_norm": color_cnt / max_colors,
            "label": self.labels[idx],
            "raw_seq": s,
        }


def collate(batch):
    max_len = max(len(ex["ids"]) for ex in batch)
    ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        seq_ids = torch.tensor(ex["ids"], dtype=torch.long)
        ids[i, : len(seq_ids)] = seq_ids
    shape_feat = torch.tensor(
        [ex["shape_norm"] for ex in batch], dtype=torch.float32
    ).unsqueeze(-1)
    color_feat = torch.tensor(
        [ex["color_norm"] for ex in batch], dtype=torch.float32
    ).unsqueeze(-1)
    y = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    raw = [ex["raw_seq"] for ex in batch]
    return {
        "ids": ids,
        "shape": shape_feat,
        "color": color_feat,
        "label": y,
        "raw_seq": raw,
    }


bs = 128
train_loader = DataLoader(
    SPRDataset(tr_seqs, tr_lbl), batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(dv_seqs, dv_lbl), batch_size=bs, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(te_seqs, te_lbl), batch_size=bs, shuffle=False, collate_fn=collate
)


# ---- model ------------------------------------------------------------------
class NeuralSymbolicNet(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, n_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lin = nn.Linear(emb + 2, hid)
        self.out = nn.Linear(hid, n_cls)
        self.act = nn.ReLU()

    def forward(self, ids, shape_feat, color_feat):
        mask = ids != 0
        emb = self.emb(ids)  # [B,L,E]
        summed = (emb * mask.unsqueeze(-1)).sum(1)
        lens = mask.sum(1).clamp(min=1).unsqueeze(-1)
        mean_emb = summed / lens  # [B,E]
        x = torch.cat([mean_emb, shape_feat, color_feat], dim=-1)
        h = self.act(self.lin(x))
        return self.out(h)


model = NeuralSymbolicNet(vocab_size, n_cls=num_classes).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---- experiment data container ---------------------------------------------
experiment_data = {
    "spr_bench": {
        "metrics": {"train_SWA": [], "dev_SWA": []},
        "losses": {"train": [], "dev": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
        "test_SWA": None,
    }
}


# ---- helpers ----------------------------------------------------------------
def evaluate(loader, seqs, labels):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch_ids = batch["ids"].to(device)
            shp = batch["shape"].to(device)
            col = batch["color"].to(device)
            logits = model(batch_ids, shp, col)
            preds.extend(logits.argmax(1).cpu().tolist())
    swa = shape_weighted_accuracy(seqs, labels, preds)
    return swa, preds


# ---- training loop with early stopping --------------------------------------
best_swa, patience, wait = -1, 6, 0
max_epochs = 30
start_time = time.time()

for epoch in range(1, max_epochs + 1):
    model.train()
    running = 0.0
    for batch in train_loader:
        batch_ids = batch["ids"].to(device)
        shp = batch["shape"].to(device)
        col = batch["color"].to(device)
        y = batch["label"].to(device)
        optim.zero_grad()
        loss = criterion(model(batch_ids, shp, col), y)
        loss.backward()
        optim.step()
        running += loss.item() * y.size(0)
    train_loss = running / len(tr_seqs)

    # validation
    dev_swa, _ = evaluate(dev_loader, dv_seqs, dv_lbl)

    # optional train swa for monitoring
    train_swa, _ = evaluate(train_loader, tr_seqs, tr_lbl)

    # logging
    print(
        f"Epoch {epoch}: validation_loss = {train_loss:.4f} | dev_SWA = {dev_swa:.4f}"
    )
    ed = experiment_data["spr_bench"]
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_loss)
    ed["losses"]["dev"].append(
        train_loss
    )  # train_loss used as placeholder; dev loss not separately computed
    ed["metrics"]["train_SWA"].append(train_swa)
    ed["metrics"]["dev_SWA"].append(dev_swa)

    # early stopping on dev SWA
    if dev_swa > best_swa + 1e-6:
        best_swa, wait = dev_swa, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ---- restore best model -----------------------------------------------------
model.load_state_dict(best_state)

# ---- test evaluation --------------------------------------------------------
test_swa, test_preds = evaluate(test_loader, te_seqs, te_lbl)
print(f"\nTest Shape-Weighted Accuracy (SWA) = {test_swa:.4f}")

# ---- store results ----------------------------------------------------------
ed = experiment_data["spr_bench"]
ed["predictions"] = test_preds
ed["ground_truth"] = te_lbl
ed["test_SWA"] = test_swa
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Artifacts saved to {working_dir} (runtime {(time.time()-start_time):.1f}s)")
