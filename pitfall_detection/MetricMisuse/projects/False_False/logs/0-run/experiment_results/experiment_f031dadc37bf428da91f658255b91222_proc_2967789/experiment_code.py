import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# --------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# --------- reproducibility -------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# --------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------- load dataset ----------
def locate_spr() -> pathlib.Path:
    guesses = [
        os.getenv("SPR_BENCH_PATH", ""),
        pathlib.Path.cwd() / "SPR_BENCH",
        pathlib.Path.home() / "SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for g in guesses:
        p = pathlib.Path(g)
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("SPR_BENCH not found.")


root = locate_spr()


def load_csv(name):
    return load_dataset(
        "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
    )


dsets = DatasetDict(
    train=load_csv("train.csv"), dev=load_csv("dev.csv"), test=load_csv("test.csv")
)
print({k: len(v) for k, v in dsets.items()})


# --------- vocab ----------
def tok(seq):
    return seq.strip().split()


all_tokens = [t for s in dsets["train"]["sequence"] for t in tok(s)]
vocab = ["<PAD>", "<UNK>"] + sorted({*all_tokens})
stoi = {w: i for i, w in enumerate(vocab)}
pad, unk = 0, 1
labels = sorted(set(dsets["train"]["label"]))
ltoi = {l: i for i, l in enumerate(labels)}


def encode(seq):
    return [stoi.get(t, unk) for t in tok(seq)]


# --------- datasets ----------
class SPRCls(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [ltoi[l] for l in split["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "ids": torch.tensor(encode(self.seqs[i]), dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


class SPRContrastive(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(encode(self.seqs[i]), dtype=torch.long)


def collate_lm(batch):
    lens = [len(x["ids"]) for x in batch]
    m = max(lens)
    ids = torch.full((len(batch), m), pad)
    labels = torch.tensor([b["label"] for b in batch])
    for i, b in enumerate(batch):
        ids[i, : lens[i]] = b["ids"]
    return {"ids": ids, "label": labels}


def collate_contrast(batch):
    lens = [len(x) for x in batch]
    m = max(lens)
    ids = torch.full((len(batch), m), pad)
    for i, tokvec in enumerate(batch):
        ids[i, : lens[i]] = tokvec
    return {"ids": ids}


train_loader_cls = DataLoader(
    SPRCls(dsets["train"]), batch_size=256, shuffle=True, collate_fn=collate_lm
)
dev_loader_cls = DataLoader(
    SPRCls(dsets["dev"]), batch_size=512, shuffle=False, collate_fn=collate_lm
)
con_loader = DataLoader(
    SPRContrastive(dsets["train"]),
    batch_size=512,
    shuffle=True,
    collate_fn=collate_contrast,
)


# --------- metrics -------------
def count_shape(seq):
    return len({tok[0] for tok in seq.strip().split()})


def count_color(seq):
    return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})


def SWA(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / sum(w)


def CWA(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / sum(w)


def SCWA(seqs, y_t, y_p):
    w = [count_shape(s) + count_color(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_t, y_p) if yt == yp) / sum(w)


# --------- model ---------------
class Encoder(nn.Module):
    def __init__(self, vsz, dim):
        super().__init__()
        self.emb = nn.Embedding(vsz, dim, padding_idx=pad)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def mean_pool(self, x):
        m = (x != pad).unsqueeze(-1)
        e = self.emb(x)
        return (e * m).sum(1) / (m.sum(1).clamp(min=1))

    def forward(self, x, project=True):
        h = self.mean_pool(x)
        return nn.functional.normalize(self.proj(h), dim=1) if project else h


class Classifier(nn.Module):
    def __init__(self, enc, dim, num_labels):
        super().__init__()
        self.enc = enc
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, x):
        h = self.enc.mean_pool(x)
        return self.fc(h)


# --------- augmentation ----------
def augment(ids):
    ids = ids.tolist()
    # token delete
    ids = [t for t in ids if t != pad and random.random() > 0.15]
    # local shuffle
    for i in range(0, len(ids) - 2, 3):
        if random.random() < 0.3:
            ids[i : i + 3] = random.sample(ids[i : i + 3], k=len(ids[i : i + 3]))
    return torch.tensor(ids if ids else [unk])


def make_views(batch_ids):
    view1 = [augment(seq) for seq in batch_ids]
    view2 = [augment(seq) for seq in batch_ids]
    lens = [len(v) for v in view1 + view2]
    m = max(lens)
    padded = torch.full((len(view1) * 2, m), pad)
    for i, v in enumerate(view1 + view2):
        padded[i, : len(v)] = v
    return padded.to(device)


# --------- contrastive pretraining -----------
def info_nce(z, temperature=0.1):
    N = z.size(0) // 2
    sim = torch.matmul(z, z.T) / temperature
    labels = torch.arange(N, device=device)
    loss = (
        nn.functional.cross_entropy(sim[:N, N:], labels)
        + nn.functional.cross_entropy(sim[N:, :N], labels)
    ) / 2
    return loss


enc_dim = 128
encoder = Encoder(len(vocab), enc_dim).to(device)
optim_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3)
contrast_epochs = 3
for ep in range(1, contrast_epochs + 1):
    encoder.train()
    tot = 0
    bs = 0
    for batch in con_loader:
        batch_ids = batch["ids"].to(device)
        optim_enc.zero_grad()
        v = make_views(batch_ids)
        z = encoder(v, project=True)
        loss = info_nce(z)
        loss.backward()
        optim_enc.step()
        tot += loss.item() * batch_ids.size(0)
        bs += batch_ids.size(0)
    print(f"[Contrast] epoch {ep} loss={tot/bs:.4f}")

# --------- fine-tune classifier ------------
classifier = Classifier(encoder, enc_dim, len(labels)).to(device)
optim_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


def evaluate(model, loader, splitname):
    model.eval()
    preds = []
    trues = []
    seqs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device)
            logits = model(ids)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(batch["label"].tolist())
            seqs.extend(
                [
                    tok
                    for tok in dsets[splitname]["sequence"][
                        len(seqs) : len(seqs) + len(ids)
                    ]
                ]
            )
    f1 = f1_score(trues, preds, average="macro")
    swa = SWA(seqs, trues, preds)
    cwa = CWA(seqs, trues, preds)
    scwa = SCWA(seqs, trues, preds)
    return f1, swa, cwa, scwa, preds, trues, seqs


ft_epochs = 5
for ep in range(1, ft_epochs + 1):
    classifier.train()
    tl = 0
    bs = 0
    for batch in train_loader_cls:
        ids = batch["ids"].to(device)
        lbl = batch["label"].to(device)
        optim_cls.zero_grad()
        logits = classifier(ids)
        loss = criterion(logits, lbl)
        loss.backward()
        optim_cls.step()
        tl += loss.item() * len(lbl)
        bs += len(lbl)
    tr_f1, _, _, _, _, _, _ = evaluate(classifier, train_loader_cls, "train")
    v_f1, v_swa, v_cwa, v_scwa, preds, trues, seqs = evaluate(
        classifier, dev_loader_cls, "dev"
    )
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((tr_f1,))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((v_f1, v_swa, v_cwa, v_scwa))
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tl / bs)
    experiment_data["SPR_BENCH"]["epochs"].append(ep)
    print(
        f"Epoch {ep}: val_loss={v_f1:.4f} F1={v_f1:.4f} SWA={v_swa:.4f} CWA={v_cwa:.4f} SCWA={v_scwa:.4f}"
    )

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = trues
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
