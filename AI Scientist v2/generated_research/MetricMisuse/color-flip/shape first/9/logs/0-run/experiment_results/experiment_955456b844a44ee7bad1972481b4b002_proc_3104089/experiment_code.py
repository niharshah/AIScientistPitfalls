import os, pathlib, random, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict as HFDD

# ------------------- working dir -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- device ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------- load data ---------------------
def load_spr_bench(root: pathlib.Path):
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = HFDD()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path(
    "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/"
)  # adjust if needed
spr = load_spr_bench(DATA_PATH)


# ------------------- helper metrics ----------------
def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ------------------- vocab & labels ----------------
vocab = {"<pad>": 0, "<unk>": 1}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


# ------------------- data augmentation -------------
def augment(seq_tokens):
    tokens = []
    for t in seq_tokens:
        if random.random() < 0.10:
            continue  # deletion
        if random.random() < 0.10:
            tokens.append("<unk>")  # mask
        else:
            tokens.append(t)
    if len(tokens) > 1 and random.random() < 0.20:  # local shuffle
        i, j = random.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    if not tokens:
        tokens = ["<unk>"]
    return tokens


# ------------------- dataset -----------------------
class SPRContrastDataset(Dataset):
    def __init__(self, split):
        self.raw_seq = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]

    def encode(self, toks):
        return [vocab.get(t, 1) for t in toks]

    def __getitem__(self, idx):
        seq_str = self.raw_seq[idx]
        toks = seq_str.split()
        view1 = self.encode(augment(toks))
        view2 = self.encode(augment(toks))
        orig = self.encode(toks)
        return {
            "orig": orig,
            "v1": view1,
            "v2": view2,
            "label": self.labels[idx],
            "raw": seq_str,
        }

    def __len__(self):
        return len(self.raw_seq)


def collate(batch):
    def pad(seqs):
        maxlen = max(len(s) for s in seqs)
        arr = np.full((len(seqs), maxlen), 0, dtype=np.int64)
        lengths = []
        for i, s in enumerate(seqs):
            arr[i, : len(s)] = s
            lengths.append(len(s))
        return torch.tensor(arr), torch.tensor(lengths)

    orig, lens_o = pad([b["orig"] for b in batch])
    v1, lens1 = pad([b["v1"] for b in batch])
    v2, lens2 = pad([b["v2"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    raws = [b["raw"] for b in batch]
    return {
        "orig": orig,
        "len_o": lens_o,
        "v1": v1,
        "len1": lens1,
        "v2": v2,
        "len2": lens2,
        "label": labels,
        "raw": raws,
    }


train_loader = DataLoader(
    SPRContrastDataset(spr["train"]), batch_size=256, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRContrastDataset(spr["dev"]), batch_size=512, shuffle=False, collate_fn=collate
)


# ------------------- model -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return mean


class SPRModel(nn.Module):
    def __init__(self, vocab_size, nclass):
        super().__init__()
        self.enc = Encoder(vocab_size)
        self.cls = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, nclass))

    def forward(self, batch, mode="orig"):
        if mode == "orig":
            return self.cls(self.enc(batch["orig"], batch["len_o"]))
        elif mode == "v1":
            return self.enc(batch["v1"], batch["len1"])
        else:
            return self.enc(batch["v2"], batch["len2"])


model = SPRModel(len(vocab), len(labels)).to(device)

# ------------------- losses & optim ---------------
ce_loss = nn.CrossEntropyLoss()
temperature = 0.5


def contrastive(z1, z2):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2B,D
    sim = torch.matmul(z, z.T) / temperature  # 2B,2B
    mask = (~torch.eye(2 * B, dtype=torch.bool, device=z.device)).float()
    sim = sim - 1e9 * (1 - mask)  # remove self-sim
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], 0)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------- logging ----------------------
experiment_data = {
    "contrastive_cls": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------- training loop ---------------
best_val_loss = float("inf")
patience = 2
wait = 0
epochs = 12
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    running = 0
    steps = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch, "orig")
        loss_cls = ce_loss(logits, batch["label"])
        z1 = model(batch, "v1")
        z2 = model(batch, "v2")
        loss_con = contrastive(z1, z2)
        loss = loss_cls + 0.1 * loss_con
        loss.backward()
        optimizer.step()
        running += loss.item() * logits.size(0)
        steps += logits.size(0)
    train_loss = running / steps
    experiment_data["contrastive_cls"]["losses"]["train"].append(train_loss)

    # ---- validate ----
    model.eval()
    val_loss = 0
    n = 0
    preds = []
    gts = []
    raws = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch, "orig")
            loss = ce_loss(out, batch["label"])
            val_loss += loss.item() * out.size(0)
            n += out.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
            raws.extend(batch["raw"])
    val_loss /= n
    experiment_data["contrastive_cls"]["losses"]["val"].append(val_loss)
    preds_str = [id2label[i] for i in preds]
    gts_str = [id2label[i] for i in gts]
    comp = complexity_weighted_accuracy(raws, gts_str, preds_str)
    experiment_data["contrastive_cls"]["metrics"]["val"].append(comp)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f} CompWA = {comp:.4f}")
    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait = 0
        experiment_data["contrastive_cls"]["predictions"] = preds_str
        experiment_data["contrastive_cls"]["ground_truth"] = gts_str
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
    if wait >= patience:
        break

# --------------- save metrics ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training finished, data saved to ./working/")
