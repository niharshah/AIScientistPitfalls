# ---------------- No-Proj ablation: full, runnable script --------------------
import os, random, math, pathlib, datetime, numpy as np, torch, warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------- folders ------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------------------- SPR benchmark loader --------------------------------
def load_spr_bench(root: pathlib.Path):
    from datasets import load_dataset

    def _load(fname):  # every csv is its own split
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    return {
        "train": _load("train.csv"),
        "dev": _load("dev.csv"),
        "test": _load("test.csv"),
    }


def try_load_spr():
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Found SPR_BENCH at", p)
            return load_spr_bench(p)
    return None


real_dset = try_load_spr()
# --------------------- synthetic fallback -----------------------------------
shapes, colors = list("RSTUVWXYZ"), list("ABCDEFGH")


def rnd_token():
    return random.choice(shapes) + random.choice(colors)


def rnd_seq():
    return " ".join(rnd_token() for _ in range(random.randint(4, 12)))


def make_split(n):
    return [
        {"id": i, "sequence": rnd_seq(), "label": random.randint(0, 3)}
        for i in range(n)
    ]


if real_dset is None:
    print("Using synthetic data")
    real_dset = {
        "train": make_split(6000),
        "dev": make_split(1200),
        "test": make_split(1200),
    }


# ------------------------- CCWA metric --------------------------------------
def count_shape_variety(s):
    return len({tok[0] for tok in s.split() if tok})


def count_color_variety(s):
    return len({tok[1] for tok in s.split() if len(tok) > 1})


def ccwa_metric(seqs, ytrue, ypred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return (
        sum(wi if t == p else 0 for wi, t, p in zip(w, ytrue, ypred)) / sum(w)
        if sum(w)
        else 0.0
    )


# ---------------------------- vocab -----------------------------------------
PAD, UNK, MASK = "<PAD>", "<UNK>", "<MASK>"


def build_vocab(rows):
    vocab = set()
    [vocab.update(r["sequence"].split()) for r in rows]
    itos = [PAD, UNK, MASK] + sorted(vocab)
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


stoi, itos = build_vocab(real_dset["train"])
vocab_size, len_max = len(stoi), 20


def encode(seq):
    ids = [stoi.get(t, stoi[UNK]) for t in seq.split()][:len_max]
    ids += [stoi[PAD]] * (len_max - len(ids))
    return ids


# -------------------------- datasets ----------------------------------------
class ContrastiveSPR(Dataset):
    def __init__(s, rows):
        s.rows = rows

    def _aug(s, ids):
        toks = [t for t in ids if t != stoi[PAD]]
        if random.random() < 0.3 and len(toks) > 1:
            toks.pop(random.randrange(len(toks)))
        if random.random() < 0.3 and len(toks) > 2:
            i = random.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        if random.random() < 0.3:
            toks += random.sample(toks, k=1)
        toks = [stoi[MASK] if random.random() < 0.15 else t for t in toks]
        return torch.tensor(encode(" ".join(itos[t] for t in toks)), dtype=torch.long)

    def __len__(s):
        return len(s.rows)

    def __getitem__(s, i):
        ids = torch.tensor(encode(s.rows[i]["sequence"]), dtype=torch.long)
        return {"v1": s._aug(ids), "v2": s._aug(ids)}


class SupervisedSPR(Dataset):
    def __init__(s, rows):
        s.rows = rows

    def __len__(s):
        return len(s.rows)

    def __getitem__(s, i):
        r = s.rows[i]
        return {
            "ids": torch.tensor(encode(r["sequence"]), dtype=torch.long),
            "label": torch.tensor(r["label"], dtype=torch.long),
            "seq": r["sequence"],
        }


# ----------------------- No-Proj encoder ------------------------------------
class TransEncoder(nn.Module):
    def __init__(s, vocab, d_model=96, nhead=6, nlayers=3):
        super().__init__()
        s.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        s.pos = nn.Parameter(torch.randn(len_max, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, batch_first=True
        )
        s.tr = nn.TransformerEncoder(layer, nlayers)
        s.pool = nn.AdaptiveAvgPool1d(1)
        # NO projection head here (ablation)

    def forward(s, x):
        x = s.emb(x.to(device)) + s.pos[: x.size(1)].unsqueeze(0)
        h = s.tr(x)  # B,L,D
        h = s.pool(h.transpose(1, 2)).squeeze(-1)  # B,D
        return torch.tanh(h)  # B,96


class Classifier(nn.Module):
    def __init__(s, enc, n_cls):
        super().__init__()
        s.enc = enc
        s.fc = nn.Linear(96, n_cls)

    def forward(s, x):
        return s.fc(s.enc(x))


# --------------------- SimCLR loss ------------------------------------------
def simclr_loss(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, 1)
    sim = torch.mm(z, z.T) / temp
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
    target = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return nn.functional.cross_entropy(sim, target)


# ------------------------ dataloaders ---------------------------------------
B_PRE, B_FT = 256, 256
pre_dl = DataLoader(
    ContrastiveSPR(real_dset["train"]), batch_size=B_PRE, shuffle=True, drop_last=True
)
train_dl = DataLoader(SupervisedSPR(real_dset["train"]), batch_size=B_FT, shuffle=True)
dev_dl = DataLoader(SupervisedSPR(real_dset["dev"]), batch_size=B_FT)

# ------------------------ bookkeeping dict ----------------------------------
experiment_data = {
    "no_proj": {
        "SPR_BENCH": {
            "metrics": {"train_CCWA": [], "val_CCWA": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# ---------------------- training pipeline -----------------------------------
encoder = TransEncoder(vocab_size).to(device)
opt_pre = torch.optim.Adam(encoder.parameters(), 1e-3)
for ep in range(1, 7):
    encoder.train()
    run = 0.0
    for b in pre_dl:
        loss = simclr_loss(encoder(b["v1"]), encoder(b["v2"]))
        opt_pre.zero_grad()
        loss.backward()
        opt_pre.step()
        run += loss.item()
    print(f"[Pre] epoch {ep}: loss {run/len(pre_dl):.4f}")

clf = Classifier(encoder, len(set(r["label"] for r in real_dset["train"]))).to(device)
opt_ft = torch.optim.Adam(clf.parameters(), 2e-3)
crit = nn.CrossEntropyLoss()


def evaluate(m, dl):
    m.eval()
    preds, gts, seqs, loss = [], [], [], 0.0
    with torch.no_grad():
        for bt in dl:
            ids, lab = bt["ids"].to(device), bt["label"].to(device)
            log = m(ids)
            loss += crit(log, lab).item()
            preds += log.argmax(1).cpu().tolist()
            gts += lab.cpu().tolist()
            seqs += bt["seq"]
    return loss / len(dl), ccwa_metric(seqs, gts, preds), preds, gts


for ep in range(1, 11):
    clf.train()
    run = 0.0
    steps = 0
    for bt in train_dl:
        ids, lab = bt["ids"].to(device), bt["label"].to(device)
        loss = crit(clf(ids), lab)
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        run += loss.item()
        steps += 1
    tr_loss = run / steps
    val_loss, ccwa, pred, gt = evaluate(clf, dev_dl)
    ts = datetime.datetime.now().isoformat()
    ed = experiment_data["no_proj"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_CCWA"].append(None)
    ed["metrics"]["val_CCWA"].append(ccwa)
    ed["predictions"].append(pred)
    ed["ground_truth"].append(gt)
    ed["timestamps"].append(ts)
    print(f"[FT] ep {ep}: train {tr_loss:.4f} | val {val_loss:.4f} | CCWA {ccwa:.4f}")

# ------------------------ save ----------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
