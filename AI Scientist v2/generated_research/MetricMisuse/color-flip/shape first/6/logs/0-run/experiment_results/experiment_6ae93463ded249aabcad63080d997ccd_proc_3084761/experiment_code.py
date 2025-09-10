import os, random, pathlib, math, time, json, csv
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------- experiment data dict --------------------
experiment_data = {
    "EPOCH_PRE_tuning": {
        "SPR": {
            "EPOCH_PRE_values": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "AIS": {"val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# -------------------- misc setup -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------- helper metrics -------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_color_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ---------------------- data -------------------------------
SPR_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")


def load_csv(split):
    fpath = SPR_PATH / f"{split}.csv"
    rows = []
    if fpath.exists():
        with open(fpath) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({"sequence": r["sequence"], "label": int(r["label"])})
    return rows


def generate_toy(n=2000):
    shapes = "ABC"
    colors = "123"
    rules = [lambda s: len(s) % 2, lambda s: (s.count("A1") + s.count("B2")) % 3]
    data = []
    for i in range(n):
        seq = " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 8))
        )
        lab = rules[i % 2](seq)
        data.append({"sequence": seq, "label": lab})
    return data


dataset = {}
for split in ["train", "dev", "test"]:
    rows = load_csv(split)
    if not rows:
        rows = generate_toy(4000 if split == "train" else 1000)
    dataset[split] = rows
print({k: len(v) for k, v in dataset.items()})

# --------------------- vocab ------------------------------
tokens = set()
for split in dataset.values():
    for r in split:
        tokens.update(r["sequence"].split())
PAD, CLS = "<PAD>", "<CLS>"
itos = [PAD, CLS] + sorted(tokens)
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)
print("Vocab size:", vocab_size)


# --------------------- augment ----------------------------
def aug_sequence(seq: str) -> str:
    toks = seq.split()
    if len(toks) > 1:
        i, j = random.sample(range(len(toks)), 2)
        toks[i], toks[j] = toks[j], toks[i]
    shift = random.randint(0, len(toks) - 1)
    toks = toks[shift:] + toks[:shift]
    return " ".join(toks)


# --------------------- dataset classes -------------------
def encode(seq, max_len=None):
    ids = [stoi[CLS]] + [stoi[t] for t in seq.split()]
    if max_len:
        ids = ids[:max_len]
        ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


class SPRContrastive(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]["sequence"]
        v1 = torch.tensor(encode(aug_sequence(seq), self.max_len))
        v2 = torch.tensor(encode(aug_sequence(seq), self.max_len))
        return v1, v2


class SPRLabelled(Dataset):
    def __init__(self, rows, max_len=20):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]["sequence"]
        ids = torch.tensor(encode(seq, self.max_len))
        label = self.rows[idx]["label"]
        return ids, torch.tensor(label), seq


# --------------------- model -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, d_model=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return h.squeeze(0)


class SPRModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(encoder.gru.hidden_size, num_classes)

    def forward(self, x):
        feat = self.enc(x)
        return self.fc(feat), feat


def nt_xent(features, temp=0.5):
    N = features.shape[0] // 2
    f = torch.nn.functional.normalize(features, dim=1)
    sim = torch.matmul(f, f.t()) / temp
    mask = torch.eye(2 * N, device=features.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=features.device)
    targets = torch.cat([targets, torch.arange(0, N, device=features.device)])
    return torch.nn.functional.cross_entropy(sim, targets)


# -------------------- loaders constant ---------------------
BATCH = 128
max_len = 20
contrast_dataset = SPRContrastive(dataset["train"], max_len)
label_train_dataset = SPRLabelled(dataset["train"], max_len)
label_dev_dataset = SPRLabelled(dataset["dev"], max_len)
contrast_loader_base = DataLoader(contrast_dataset, batch_size=BATCH, shuffle=True)
train_loader_base = DataLoader(label_train_dataset, batch_size=BATCH, shuffle=True)
dev_loader = DataLoader(label_dev_dataset, batch_size=BATCH)
num_classes = len(set(r["label"] for r in dataset["train"]))
EPOCH_FT = 3  # keep fine-tune constant

# -------------------- hyperparam candidates -----------------
epoch_pre_candidates = [3, 8, 12]

for EPOCH_PRE in epoch_pre_candidates:
    print("\n============================")
    print(f"Running experiment with EPOCH_PRE = {EPOCH_PRE}")
    # fresh model
    encoder = Encoder(vocab_size).to(device)
    model = SPRModel(encoder, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ----- contrastive pre-training -----
    pre_losses = []
    model.train()
    for ep in range(1, EPOCH_PRE + 1):
        total_loss = 0
        for v1, v2 in contrast_loader_base:
            v1, v2 = v1.to(device), v2.to(device)
            _, f1 = model(v1)
            _, f2 = model(v2)
            loss = nt_xent(torch.cat([f1, f2], 0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * v1.size(0)
        l = total_loss / len(dataset["train"])
        pre_losses.append(l)
        print(f"  Pretrain epoch {ep}/{EPOCH_PRE}  loss={l:.4f}")

    # ----- supervised fine-tuning -----
    criterion = nn.CrossEntropyLoss()
    finetune_losses, swa_list, cwa_list, ais_list = [], [], [], []
    for ep in range(1, EPOCH_FT + 1):
        # train
        model.train()
        total = 0
        for ids, labels, _ in train_loader_base:
            ids, labels = ids.to(device), labels.to(device)
            logits, _ = model(ids)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * ids.size(0)
        train_loss = total / len(dataset["train"])
        finetune_losses.append(train_loss)

        # validation
        model.eval()
        val_loss, preds, gts, seqs = 0, [], [], []
        with torch.no_grad():
            for ids, labels, seq in dev_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits, _ = model(ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * ids.size(0)
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                gts.extend(labels.cpu().tolist())
                seqs.extend(seq)
        val_loss /= len(dataset["dev"])
        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)

        # AIS
        def compute_ais(rows, n_views=3):
            consistent = 0
            with torch.no_grad():
                for r in rows:
                    base = None
                    ok = True
                    for _ in range(n_views):
                        ids = (
                            torch.tensor(encode(aug_sequence(r["sequence"]), max_len))
                            .unsqueeze(0)
                            .to(device)
                        )
                        logit, _ = model(ids)
                        pred = torch.argmax(logit, 1).item()
                        if base is None:
                            base = pred
                        elif pred != base:
                            ok = False
                            break
                    if ok:
                        consistent += 1
            return consistent / len(rows)

        ais = compute_ais(dataset["dev"])
        swa_list.append(swa)
        cwa_list.append(cwa)
        ais_list.append(ais)
        print(
            f"  Finetune epoch {ep}/{EPOCH_FT} | val_loss={val_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} AIS={ais:.3f}"
        )

    # --------- store results ---------------
    exp = experiment_data["EPOCH_PRE_tuning"]["SPR"]
    exp["EPOCH_PRE_values"].append(EPOCH_PRE)
    exp["losses"]["train"].append(pre_losses + finetune_losses)
    exp["losses"]["val"].append(None)  # placeholder for symmetry
    exp["metrics"]["train"].append(swa_list)  # SWA tracked per ft epoch
    exp["metrics"]["val"].append(cwa_list)  # CWA per ft epoch
    exp["AIS"]["val"].append(ais_list[-1])
    exp["predictions"].append(preds)
    exp["ground_truth"].append(gts)

    # free cuda
    del model, encoder, optimizer
    torch.cuda.empty_cache()

# -------------------- save ------------------------------
out_path = os.path.join(working_dir, "experiment_data.npy")
np.save(out_path, experiment_data)
print("All experiments done. Saved results to", out_path)
