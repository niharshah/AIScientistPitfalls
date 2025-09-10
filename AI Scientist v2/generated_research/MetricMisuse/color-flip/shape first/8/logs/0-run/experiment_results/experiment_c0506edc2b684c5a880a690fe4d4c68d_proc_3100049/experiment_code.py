import os, random, time, datetime, pathlib, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------
#  0. housekeeping
# --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------
#  1. try to load SPR_BENCH, else synthesize toy data
# --------------------------------------------------------------------
def try_load_spr():
    try:
        from SPR import load_spr_bench

        DATA_PATH = pathlib.Path("./SPR_BENCH")
        if not DATA_PATH.exists():
            DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dsets = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH.")
        return {k: dsets[k] for k in ["train", "dev", "test"]}
    except Exception as e:
        print("Could not load SPR_BENCH, reason:", e)
        return None


real_dset = try_load_spr()


def make_random_token():
    return random.choice(list("RSTUV")) + random.choice(list("ABCDE"))


def gen_sequence():
    return " ".join(make_random_token() for _ in range(random.randint(3, 12)))


def gen_split(n):
    return [
        {"id": i, "sequence": gen_sequence(), "label": random.randint(0, 3)}
        for i in range(n)
    ]


if real_dset is None:
    print("Generating synthetic fallback data …")
    real_dset = {
        "train": gen_split(2000),
        "dev": gen_split(400),
        "test": gen_split(400),
    }


# --------------------------------------------------------------------
#  2. utility metrics
# --------------------------------------------------------------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split()})


def ccwa_metric(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / sum(weights) if sum(weights) else 0.0


# --------------------------------------------------------------------
#  3. vocab and encoding helpers
# --------------------------------------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    for r in rows:
        vocab.update(r["sequence"].split())
    lst = [PAD, UNK, MASK] + sorted(vocab)
    return {tok: i for i, tok in enumerate(lst)}, (
        {
            i: tok
            for tok, i in build_vocab.__annotations__.get("return", (None, None))[
                0
            ].items()
        }
        if False
        else {i: t for i, t in enumerate(lst)}
    )


stoi, itos = build_vocab(real_dset["train"])
vocab_size = len(stoi)
MAX_LEN = 20


def encode(seq):
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq.split()][:MAX_LEN]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


# --------------------------------------------------------------------
#  4. datasets
# --------------------------------------------------------------------
class ContrastiveMLMDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def _augment(self, ids):
        tids = [i for i in ids if i != stoi[PAD]]
        if len(tids) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(tids)), 2)
            tids[i], tids[j] = tids[j], tids[i]
        tids = [stoi[MASK] if random.random() < 0.15 else t for t in tids]
        tids = tids + [stoi[PAD]] * (MAX_LEN - len(tids))
        return torch.tensor(tids)

    def _mlm(self, ids):
        inp, lab = [], []
        for t in ids:
            if t != stoi[PAD] and random.random() < 0.15:
                inp.append(stoi[MASK])
                lab.append(t)
            else:
                inp.append(t)
                lab.append(-100)
        return torch.tensor(inp), torch.tensor(lab)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]["sequence"]
        ids = encode(seq)
        v1 = self._augment(ids)
        v2 = self._augment(ids)
        mlm_inp, mlm_lab = self._mlm(ids)
        return {"view1": v1, "view2": v2, "mlm_inp": mlm_inp, "mlm_lab": mlm_lab}


class SupervisedDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "ids": torch.tensor(encode(r["sequence"])),
            "label": torch.tensor(r["label"]),
            "seq": r["sequence"],
        }


# --------------------------------------------------------------------
#  5. model
# --------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hid * 2, 128)
        self.mlm_head = nn.Linear(hid * 2, vocab_sz, bias=False)  # tie if wanted

    def _encode(self, x):
        h, _ = self.lstm(self.emb(x))
        return h

    def pooled(self, x):
        h = self._encode(x)  # B,L,H
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)
        return torch.tanh(self.proj(pooled))  # B,128

    def mlm_logits(self, x):
        h = self._encode(x)
        return self.mlm_head(h)  # B,L,V


class Classifier(nn.Module):
    def __init__(self, encoder, num_cls):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, num_cls)

    def forward(self, x):
        return self.head(self.encoder.pooled(x))


# --------------------------------------------------------------------
#  6. losses
# --------------------------------------------------------------------
def simclr_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], 0)
    logits = torch.matmul(reps, reps.T) / temp
    logits.fill_diagonal_(-9e15)
    B = z1.size(0)
    targets = torch.arange(B, 2 * B, device=z1.device)
    targets = torch.cat([targets, torch.arange(0, B, device=z1.device)])
    return nn.functional.cross_entropy(logits, targets)


# --------------------------------------------------------------------
#  7. training / evaluation helpers
# --------------------------------------------------------------------
def evaluate(model, dl, criterion):
    model.eval()
    preds, labs, seqs = [], [], []
    loss_tot, n = 0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["ids"])
            loss = criterion(logits, batch["label"])
            loss_tot += loss.item()
            n += 1
            preds.extend(logits.argmax(1).cpu().tolist())
            labs.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["seq"])
    return loss_tot / n if n else 0.0, ccwa_metric(seqs, labs, preds), preds, labs, seqs


# --------------------------------------------------------------------
#  8. experiment dict
# --------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_CCWA": [], "val_CCWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}

# --------------------------------------------------------------------
#  9. config
# --------------------------------------------------------------------
BATCH = 256
PRE_EPOCHS = 2
FT_EPOCHS = 4
NUM_CLASSES = len({r["label"] for r in real_dset["train"]})
print(f"Vocab size = {vocab_size}, Num classes = {NUM_CLASSES}")

# --------------------------------------------------------------------
# 10. dataloaders
# --------------------------------------------------------------------
pre_ds = ContrastiveMLMDataset(real_dset["train"])
pre_dl = DataLoader(pre_ds, batch_size=BATCH, shuffle=True, drop_last=True)

train_ds = SupervisedDataset(real_dset["train"])
dev_ds = SupervisedDataset(real_dset["dev"])
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
dev_dl = DataLoader(dev_ds, batch_size=BATCH, shuffle=False)

# --------------------------------------------------------------------
# 11. pre-training
# --------------------------------------------------------------------
encoder = Encoder(vocab_size).to(device)
opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

for ep in range(1, PRE_EPOCHS + 1):
    encoder.train()
    t0 = time.time()
    tot = 0
    n = 0
    for batch in pre_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        z1, z2 = encoder.pooled(batch["view1"]), encoder.pooled(batch["view2"])
        loss_c = simclr_loss(z1, z2)
        mlm_logits = encoder.mlm_logits(batch["mlm_inp"])
        loss_mlm = mlm_criterion(
            mlm_logits.view(-1, vocab_size), batch["mlm_lab"].view(-1)
        )
        loss = loss_c + loss_mlm
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    print(f"Pre-Epoch {ep}: loss={tot/n:.4f}  time={time.time()-t0:.1f}s")

# --------------------------------------------------------------------
# 12. fine-tuning
# --------------------------------------------------------------------
clf = Classifier(encoder, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
opt_ft = torch.optim.Adam(clf.parameters(), lr=2e-3)

for ep in range(1, FT_EPOCHS + 1):
    clf.train()
    tot = 0
    n = 0
    for batch in train_dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss = criterion(clf(batch["ids"]), batch["label"])
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        tot += loss.item()
        n += 1
    train_loss = tot / n
    val_loss, val_ccwa, preds, gts, seqs = evaluate(clf, dev_dl, criterion)
    ts = datetime.datetime.now().isoformat()
    # logging
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_CCWA"].append(None)
    experiment_data["SPR_BENCH"]["metrics"]["val_CCWA"].append(val_ccwa)
    experiment_data["SPR_BENCH"]["predictions"].append(preds)
    experiment_data["SPR_BENCH"]["ground_truth"].append(gts)
    experiment_data["SPR_BENCH"]["timestamps"].append(ts)
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f} | CCWA = {val_ccwa:.4f}")

# --------------------------------------------------------------------
# 13. save everything
# --------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
