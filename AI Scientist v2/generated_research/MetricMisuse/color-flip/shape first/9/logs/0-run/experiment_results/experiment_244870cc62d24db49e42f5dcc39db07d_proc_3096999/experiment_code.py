import os, random, pathlib, time, math, json, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------
# mandatory working directory & device announcement
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ---------------------------------------------------


# ------------- helper functions & metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum((wt if t == p else 0) for wt, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------------- dataset loading ------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):
        return load_dataset(
            "csv", data_files=str(root / fname), split="train", cache_dir=".cache_dsets"
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"] = _load("dev.csv")
    dset["test"] = _load("test.csv")
    return dset


def generate_synthetic(path: str):
    os.makedirs(path, exist_ok=True)

    def rand_seq():
        shapes, colors = "ABCDE", "12345"
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def make_csv(fname, n):
        with open(os.path.join(path, fname), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = rand_seq()
                lbl = int(len(seq.replace(" ", "")[::2]) % 3)  # simple rule
                f.write(f"{i},{seq},{lbl}\n")

    make_csv("train.csv", 2000)
    make_csv("dev.csv", 400)
    make_csv("test.csv", 600)


data_root = pathlib.Path(os.getcwd()) / "SPR_BENCH"
if not data_root.exists():
    print("SPR_BENCH not found → generating tiny synthetic data.")
    generate_synthetic(str(data_root))

dsets = load_spr_bench(data_root)
print({k: len(v) for k, v in dsets.items()})

# --------------- vocabulary & mapping --------------
PAD, UNK = 0, 1
token2id = {"<PAD>": PAD, "<UNK>": UNK}


def build_vocab(dataset):
    for seq in dataset["sequence"]:
        for tok in seq.split():
            if tok not in token2id:
                token2id[tok] = len(token2id)


build_vocab(dsets["train"])
vocab_size = len(token2id)
print("Vocab", vocab_size)

label2id = {}
for lbl in dsets["train"]["label"]:
    if lbl not in label2id:
        label2id[lbl] = len(label2id)
num_classes = len(label2id)


def encode_sequence(seq):
    return [token2id.get(tok, UNK) for tok in seq.split()]


# ------------------ torch Dataset ------------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2id[lbl] for lbl in hf_split["label"]]
        self.ids = hf_split["id"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": encode_sequence(self.seqs[idx]),
            "label": self.labels[idx],
            "raw": self.seqs[idx],
        }


def pad_batch(items):
    maxlen = max(len(x["input_ids"]) for x in items)
    input_ids = []
    labels = []
    raws = []
    for it in items:
        arr = it["input_ids"] + [PAD] * (maxlen - len(it["input_ids"]))
        input_ids.append(arr)
        labels.append(it["label"])
        raws.append(it["raw"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "raw": raws,
    }


train_loader = DataLoader(
    SPRTorchDataset(dsets["train"]), batch_size=128, shuffle=True, collate_fn=pad_batch
)
dev_loader = DataLoader(
    SPRTorchDataset(dsets["dev"]), batch_size=256, shuffle=False, collate_fn=pad_batch
)
test_loader = DataLoader(
    SPRTorchDataset(dsets["test"]), batch_size=256, shuffle=False, collate_fn=pad_batch
)


# --------------- augmentation ----------------------
def dropout_tokens(tokens, p=0.15):
    keep = [tok for tok in tokens if random.random() > p]
    if not keep:
        keep = [random.choice(tokens)]
    return keep


def augment_batch(seqs):
    return [dropout_tokens(s) for s in seqs]


# ------------------- model -------------------------
class SPRModel(nn.Module):
    def __init__(self, vocab, n_classes, emb=128, hid=128, proj=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hid * 2, n_classes)
        self.projector = nn.Sequential(
            nn.Linear(hid * 2, proj), nn.ReLU(), nn.Linear(proj, proj)
        )

    def forward(self, x):
        mask = x != PAD
        emb = self.embed(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, enforce_sorted=False, batch_first=True
        )
        _, h = self.encoder(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # B x 2H
        logits = self.classifier(h)
        z = F.normalize(self.projector(h), dim=-1)
        return logits, z


model = SPRModel(vocab_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------ SimCLR loss --------------------------
def nt_xent(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D (already normalized)
    sim = (z @ z.T) / temp
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets])
    loss = F.cross_entropy(sim, targets)
    return loss


# ------------- experiment data dict ---------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------- training ----------------------
epochs = 5
alpha = 0.5
for epoch in range(1, epochs + 1):
    model.train()
    tr_loss = 0
    for batch in train_loader:
        ids = batch["input_ids"]
        labels = batch["labels"]
        orig = ids.to(device)
        labels = labels.to(device)
        # build two augmented views
        seq_list = [seq.tolist() for seq in ids]
        view1 = [
            encode_sequence(
                " ".join(
                    [list(token2id.keys())[tok] for tok in dropout_tokens(list_tokens)]
                )
            )
            for list_tokens in seq_list
        ]
        view2 = [
            encode_sequence(
                " ".join(
                    [list(token2id.keys())[tok] for tok in dropout_tokens(list_tokens)]
                )
            )
            for list_tokens in seq_list
        ]

        def pad(v):
            ml = max(len(x) for x in v)
            return torch.tensor(
                [x + [PAD] * (ml - len(x)) for x in v], dtype=torch.long
            ).to(device)

        v1 = pad(view1)
        v2 = pad(view2)

        logits, _ = model(orig)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss_cls = F.cross_entropy(logits, labels)
        loss_con = nt_xent(z1, z2)
        loss = loss_cls + alpha * loss_con
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

    # -------- validation -------------
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_seqs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(ids)
            val_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())
            all_seqs.extend(batch["raw"])
    val_loss /= len(dev_loader.dataset)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa2 = complexity_weighted_accuracy(all_seqs, all_labels, all_preds)
    metrics = {"swa": swa, "cwa": cwa, "cwa2d": cwa2}
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(metrics)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA {swa:.3f} CWA {cwa:.3f} CWA-2D {cwa2:.3f}"
    )

# -------- final test evaluation --------------------
model.eval()
test_preds = []
test_labels = []
test_seqs = []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        logits, _ = model(ids)
        test_preds.extend(logits.argmax(-1).cpu().tolist())
        test_labels.extend(batch["labels"].tolist())
        test_seqs.extend(batch["raw"])
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_labels
experiment_data["SPR_BENCH"]["metrics"]["test"] = {
    "swa": shape_weighted_accuracy(test_seqs, test_labels, test_preds),
    "cwa": color_weighted_accuracy(test_seqs, test_labels, test_preds),
    "cwa2d": complexity_weighted_accuracy(test_seqs, test_labels, test_preds),
}
print("Test metrics:", experiment_data["SPR_BENCH"]["metrics"]["test"])

# -------------- save experiment data ---------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
