import os, random, time, pathlib, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# ------------------------------------------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------------------------
# reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ------------------------------------------------------------------------------------
# metric helpers from the benchmark paper
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) + 1e-9)


# ------------------------------------------------------------------------------------
# data loading (real or synthetic fallback)
def try_load_spr_bench(root: pathlib.Path):
    try:
        from datasets import load_dataset

        def _ld(name):
            return load_dataset(
                "csv",
                data_files=str(root / name),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        return True, d
    except Exception as e:
        print("Could not load SPR_BENCH – using synthetic toy data.", e)
        return False, {}


def make_synth_dataset(n):
    shapes, colors = list("ABCDE"), list("12345")
    seqs, labels = [], []
    for _ in range(n):
        L = random.randint(4, 9)
        seq = " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))
        seqs.append(seq)
        labels.append(int(count_shape_variety(seq) >= count_color_variety(seq)))
    return {"sequence": seqs, "label": labels}


SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
have_real, raw = try_load_spr_bench(SPR_PATH)
if have_real:
    train_raw = {"sequence": raw["train"]["sequence"], "label": raw["train"]["label"]}
    dev_raw = {"sequence": raw["dev"]["sequence"], "label": raw["dev"]["label"]}
    test_raw = {"sequence": raw["test"]["sequence"], "label": raw["test"]["label"]}
else:
    train_raw = make_synth_dataset(2000)
    dev_raw = make_synth_dataset(400)
    test_raw = make_synth_dataset(400)

# ------------------------------------------------------------------------------------
# vocabulary
special = ["<pad>", "<unk>", "<mask>"]
all_toks = sorted({tok for seq in train_raw["sequence"] for tok in seq.split()})
vocab = {tok: i + len(special) for i, tok in enumerate(all_toks)}
for i, s in enumerate(special):
    vocab[s] = i
pad_idx, unk_idx, mask_idx = vocab["<pad>"], vocab["<unk>"], vocab["<mask>"]


def encode(tokens: List[str], max_len: int):
    ids = [vocab.get(t, unk_idx) for t in tokens][:max_len]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


max_len = max(len(s.split()) for s in train_raw["sequence"])


# ------------------------------------------------------------------------------------
# Dataset wrappers
class SPRDataset(Dataset):
    def __init__(self, seqs, labels=None):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        item = {"seq": self.seqs[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


train_ds = SPRDataset(train_raw["sequence"], train_raw["label"])
dev_ds = SPRDataset(dev_raw["sequence"], dev_raw["label"])
test_ds = SPRDataset(test_raw["sequence"], test_raw["label"])


def collate(batch):
    seqs = [b["seq"] for b in batch]
    ids = torch.tensor([encode(s.split(), max_len) for s in seqs], dtype=torch.long)
    out = {"x": ids, "raw": seqs}
    if "label" in batch[0]:
        out["y"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, collate_fn=collate)


# ------------------------------------------------------------------------------------
# simple masking augmentation
def augment(seq: str, p_mask=0.15) -> str:
    toks = seq.split()
    for i, t in enumerate(toks):
        if random.random() < p_mask:
            toks[i] = "<mask>"
    return " ".join(toks)


# ------------------------------------------------------------------------------------
# model
class GRUEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, num_layers=layers)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return h[-1]  # (B, hid)


class Classifier(nn.Module):
    def __init__(self, encoder, hid_dim, n_classes):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(hid_dim, n_classes)

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        z = self.enc(x)
        return self.fc(z)


# Contrastive loss (NT-Xent)
def nt_xent(z1, z2, temp=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # (2B,D)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temp  # (2B,2B)
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    pos_sim = sim[torch.arange(2 * B), pos]
    denom = torch.logsumexp(sim, dim=1)
    loss = (-pos_sim + denom).mean()
    return loss


# ------------------------------------------------------------------------------------
# experiment bookkeeping
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"val_SWA": [], "val_ACA": [], "val_loss": []},
        "losses": {"pretrain": [], "finetune": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ------------------------------------------------------------------------------------
# build model
hid_dim, emb_dim, layers = 128, 64, 1
enc = GRUEncoder(len(vocab), emb_dim, hid_dim, layers).to(device)
model = Classifier(enc, hid_dim, n_classes=len(set(train_raw["label"]))).to(device)

# ------------------------------------------------------------------------------------
# 1. contrastive pre-training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS_PRE = 3
for ep in range(1, EPOCHS_PRE + 1):
    model.train()
    tot_loss, n = 0, 0
    for batch in train_loader:
        seqs = batch["raw"]
        aug1 = [augment(s) for s in seqs]
        aug2 = [augment(s) for s in seqs]
        x1 = torch.tensor(
            [encode(s.split(), max_len) for s in aug1], dtype=torch.long
        ).to(device)
        x2 = torch.tensor(
            [encode(s.split(), max_len) for s in aug2], dtype=torch.long
        ).to(device)
        z1, z2 = model.encode(x1), model.encode(x2)
        loss = nt_xent(z1, z2, 0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * x1.size(0)
        n += x1.size(0)
    avg_loss = tot_loss / n
    experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(avg_loss)
    print(f"Contrastive epoch {ep}: loss={avg_loss:.4f}")

# ------------------------------------------------------------------------------------
# 2. supervised fine-tuning
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS_FT = 5


def evaluate(loader):
    model.eval()
    tots, raws, gts, preds = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            tots += loss.item() * y.size(0)
            pr = logits.argmax(1).cpu().tolist()
            preds.extend(pr)
            gts.extend(batch["y"].cpu().tolist())
            raws.extend(batch["raw"])
    return (
        tots / len(loader.dataset),
        shape_weighted_accuracy(raws, gts, preds),
        preds,
        gts,
        raws,
    )


def compute_aca(raws, labels, M=3):
    model.eval()
    correct = 0
    tot = len(raws) * (M + 1)
    with torch.no_grad():
        for seq, label in zip(raws, labels):
            seqs = [seq] + [augment(seq) for _ in range(M)]
            xs = torch.tensor(
                [encode(s.split(), max_len) for s in seqs], dtype=torch.long
            ).to(device)
            logits = model(xs)
            preds = logits.argmax(1).cpu().tolist()
            correct += sum(int(p == label) for p in preds)
    return correct / tot


for ep in range(1, EPOCHS_FT + 1):
    model.train()
    tr_loss, n = 0, 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch["y"].size(0)
        n += batch["y"].size(0)
    tr_loss /= n
    val_loss, val_swa, _, _, val_raws = evaluate(dev_loader)
    val_aca = compute_aca(val_raws, dev_raw["label"])
    experiment_data["SPR_BENCH"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val_SWA"].append(val_swa)
    experiment_data["SPR_BENCH"]["metrics"]["val_ACA"].append(val_aca)
    experiment_data["SPR_BENCH"]["losses"]["finetune"].append(tr_loss)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | val_SWA={val_swa:.4f} | val_ACA={val_aca:.4f}"
    )

# ------------------------------------------------------------------------------------
# 3. final test evaluation
test_loss, test_swa, test_preds, test_gts, test_raws = evaluate(test_loader)
test_aca = compute_aca(test_raws, test_raw["label"])
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
print(f"Test: loss={test_loss:.4f} | SWA={test_swa:.4f} | ACA={test_aca:.4f}")

# ------------------------------------------------------------------------------------
# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
