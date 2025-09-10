import os, random, itertools, time, pathlib, warnings, math, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------- mandatory working folder -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- reproducibility & device -----------------
warnings.filterwarnings("ignore")
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- experiment dict --------------------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {
            "train_SWA": [],
            "val_SWA": [],
            "train_CWA": [],
            "val_CWA": [],
            "train_ACR": [],
            "val_ACR": [],
            "test_SWA": [],
            "test_CWA": [],
            "test_ACR": [],
        },
        "predictions": [],
        "ground_truth": [],
    }
}


# ----------------- helpers to load (or mock) dataset --------
def safe_load_spr():
    """
    Try to load the real SPR_BENCH folder via SPR.py; otherwise fabricate
    a tiny synthetic benchmark so that the script is runnable everywhere.
    """
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("SPR_BENCH")
        if DATA_PATH.exists():
            d = load_spr_bench(DATA_PATH)
            print("Loaded real SPR_BENCH.")
            return {k: [dict(r) for r in d[k]] for k in d}
    except Exception as e:
        print("Could not load real SPR_BENCH – falling back to synthetic toy set.", e)

    shapes, colors = list("ABCDEFG"), list("123456")

    def token():
        return random.choice(shapes) + random.choice(colors)

    def rand_seq():
        return " ".join(token() for _ in range(random.randint(4, 10)))

    def label_fn(s):
        return 1 if sum(tok[0] in "ABC" for tok in s.split()) % 2 == 0 else 0

    toy = {sp: [] for sp in ["train", "dev", "test"]}
    for split, n in [("train", 2000), ("dev", 400), ("test", 400)]:
        for i in range(n):
            seq = rand_seq()
            toy[split].append({"id": i, "sequence": seq, "label": label_fn(seq)})
    return toy


dset = safe_load_spr()
print({k: len(v) for k, v in dset.items()})

# ----------------- vocabulary & encoding --------------------
PAD, MASK, UNK = "<PAD>", "<MASK>", "<UNK>"


def build_vocab(seqs):
    vocab = [PAD, MASK, UNK] + sorted(
        set(itertools.chain.from_iterable(s.split() for s in seqs))
    )
    return vocab, {tok: i for i, tok in enumerate(vocab)}


vocab, stoi = build_vocab([r["sequence"] for r in dset["train"]])
itos = {i: s for s, i in stoi.items()}


def encode(seq):
    return [stoi.get(t, stoi[UNK]) for t in seq.split()]


# ----------------- metrics as defined by benchmark ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ----------------- dataset & collators ----------------------
class SPRDataset(Dataset):
    def __init__(self, records, with_label=True):
        self.records, self.with_label = records, with_label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        out = {
            "input_ids": torch.tensor(encode(rec["sequence"]), dtype=torch.long),
            "sequence": rec["sequence"],
        }
        if self.with_label:
            out["label"] = torch.tensor(rec["label"], dtype=torch.long)
        return out


def pad_and_stack(seqs, max_len, dtype=torch.long):
    out = torch.zeros(len(seqs), max_len, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate_classification(batch):
    ids = [b["input_ids"] for b in batch]
    lens = [len(x) for x in ids]
    padded = pad_and_stack(ids, max(lens))
    res = {
        "input_ids": padded.to(device),
        "lengths": torch.tensor(lens).to(device),
        "sequence": [b["sequence"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]).to(device),
    }
    return res


def augment_ids(id_list):
    ids = id_list.copy()
    for i in range(len(ids)):
        if random.random() < 0.15:
            ids[i] = stoi[MASK]
    for i in range(len(ids) - 1):
        if random.random() < 0.1:
            ids[i], ids[i + 1] = ids[i + 1], ids[i]
    if len(ids) > 4 and random.random() < 0.3:
        del ids[random.randint(0, len(ids) - 1)]
    return ids


def collate_contrastive(batch):
    bases = [b["input_ids"] for b in batch]
    views = []
    for base in bases:
        views.append(torch.tensor(augment_ids(base.tolist()), dtype=torch.long))
        views.append(torch.tensor(augment_ids(base.tolist()), dtype=torch.long))
    lens = [len(v) for v in views]
    padded = pad_and_stack(views, max(lens))
    return {"input_ids": padded.to(device), "lengths": torch.tensor(lens).to(device)}


# ----------------- model ------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(d_model * 2, d_model)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        pooled = (h * mask).sum(1) / (mask.sum(1) + 1e-6)
        return self.lin(pooled)


class SPRModel(nn.Module):
    def __init__(self, vocab_sz, num_labels):
        super().__init__()
        self.encoder = Encoder(vocab_sz)
        self.projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x, lengths, use_projection=False):
        rep = self.encoder(x, lengths)
        if use_projection:
            return self.projection(rep)
        return self.classifier(rep), rep


# ----------------- NT-Xent loss ------------------------------
def nt_xent(z, temperature=0.5):
    N = z.size(0) // 2
    z = F.normalize(z, dim=1)
    sim = torch.exp(z @ z.t() / temperature)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim * mask
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], 0)
    denom = sim.sum(1)
    return (-torch.log(pos / denom)).mean()


# ----------------- dataloaders ------------------------------
batch_c = 128
train_con_loader = DataLoader(
    SPRDataset(dset["train"], with_label=False),
    batch_size=batch_c // 2,
    shuffle=True,
    collate_fn=collate_contrastive,
    drop_last=True,
)
train_loader = DataLoader(
    SPRDataset(dset["train"]),
    batch_size=64,
    shuffle=True,
    collate_fn=collate_classification,
)
val_loader = DataLoader(
    SPRDataset(dset["dev"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)
test_loader = DataLoader(
    SPRDataset(dset["test"]),
    batch_size=64,
    shuffle=False,
    collate_fn=collate_classification,
)

# ----------------- model init & optimizers ------------------
model = SPRModel(len(vocab), num_labels=len(set(r["label"] for r in dset["train"]))).to(
    device
)
opt_contrastive = torch.optim.Adam(
    list(model.encoder.parameters()) + list(model.projection.parameters()), lr=1e-3
)

# ----------------- contrastive pre-training -----------------
epochs_ct = 3
for ep in range(1, epochs_ct + 1):
    model.train()
    losses = []
    for batch in train_con_loader:
        z = model(batch["input_ids"], batch["lengths"], use_projection=True)
        loss = nt_xent(z)
        opt_contrastive.zero_grad()
        loss.backward()
        opt_contrastive.step()
        losses.append(loss.item())
    print(f"[Pretrain] epoch {ep}/{epochs_ct} loss={np.mean(losses):.4f}")

# ----------------- fine-tuning classifier -------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def compute_acr(seqs, preds_fn, k=3):
    """
    Augmentation Consistency Rate: fraction whose predictions remain identical
    over k augmented views.
    """
    consistent = 0
    with torch.no_grad():
        for s in seqs:
            base_ids = torch.tensor(encode(s), dtype=torch.long).unsqueeze(0).to(device)
            base_len = torch.tensor([base_ids.size(1)]).to(device)
            base_pred = preds_fn(base_ids, base_len)
            ok = True
            for _ in range(k):
                aug = (
                    torch.tensor(augment_ids(encode(s)), dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )
                alen = torch.tensor([aug.size(1)]).to(device)
                if preds_fn(aug, alen) != base_pred:
                    ok = False
                    break
            consistent += int(ok)
    return consistent / len(seqs) if seqs else 0.0


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_truths, all_seqs = 0.0, [], [], []
    for batch in loader:
        logits, _ = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(1).cpu().numpy()
        truths = batch["label"].cpu().numpy()
        all_preds.extend(preds)
        all_truths.extend(truths)
        all_seqs.extend(batch["sequence"])
    swa = shape_weighted_accuracy(all_seqs, all_truths, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_truths, all_preds)
    acr = compute_acr(
        all_seqs, lambda x, l: model(x, l)[0].argmax(1).item(), k=2
    )  # k=2 for speed
    return (
        total_loss / len(loader.dataset),
        swa,
        cwa,
        acr,
        all_preds,
        all_truths,
        all_seqs,
    )


best_val_swa = 0
clf_epochs = 5
for ep in range(1, clf_epochs + 1):
    tr_loss, tr_swa, tr_cwa, tr_acr, *_ = run_epoch(train_loader, True)
    val_loss, val_swa, val_cwa, val_acr, *_ = run_epoch(val_loader, False)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_SWA"].append(tr_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_SWA"].append(val_swa)
    experiment_data["SPR_BENCH"]["metrics"]["train_CWA"].append(tr_cwa)
    experiment_data["SPR_BENCH"]["metrics"]["val_CWA"].append(val_cwa)
    experiment_data["SPR_BENCH"]["metrics"]["train_ACR"].append(tr_acr)
    experiment_data["SPR_BENCH"]["metrics"]["val_ACR"].append(val_acr)

    print(
        f"Epoch {ep}: "
        f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
        f"val_SWA={val_swa:.4f} val_CWA={val_cwa:.4f} val_ACR={val_acr:.4f}"
    )

    best_val_swa = max(best_val_swa, val_swa)

# ----------------- final test evaluation --------------------
test_loss, test_swa, test_cwa, test_acr, preds, trues, seqs = run_epoch(
    test_loader, False
)
print(f"Test: SWA={test_swa:.4f}  CWA={test_cwa:.4f}  ACR={test_acr:.4f}")

exp = experiment_data["SPR_BENCH"]
exp["metrics"]["test_SWA"].append(test_swa)
exp["metrics"]["test_CWA"].append(test_cwa)
exp["metrics"]["test_ACR"].append(test_acr)
exp["predictions"] = preds
exp["ground_truth"] = trues

# ----------------- save everything --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", working_dir)
