import os, pathlib, random, time, copy, math, json, itertools
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# -------------------------------------------------
# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------
# Helper ------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# -------------------------------------------------
# Dataset ---------------------------------------------------------------
def find_or_create_dataset() -> DatasetDict:
    data_root_candidates = [pathlib.Path("SPR_BENCH"), pathlib.Path("./data/SPR_BENCH")]
    for p in data_root_candidates:
        if (p / "train.csv").exists():
            print(f"Loaded SPR_BENCH from {p}")
            return load_spr_bench(p)

    print("SPR_BENCH not found – generating synthetic toy data.")
    shapes, colors = "ABCD", "1234"

    def synth(n: int):
        seqs, labels = [], []
        for idx in range(n):
            L = random.randint(4, 9)
            seqs.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    ds_dict = DatasetDict()
    for split, n in [("train", 600), ("dev", 150), ("test", 100)]:
        ds_dict[split] = HFDataset.from_dict(synth(n))
    return ds_dict


spr_bench = find_or_create_dataset()


def tokenize(seq: str):
    return seq.strip().split()


# Build vocabulary
vocab = {"<PAD>": 0}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
pad_id = vocab["<PAD>"]
print(f"Vocabulary size = {vocab_size}")


# -------------------------------------------------
# PyTorch datasets -------------------------------------------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    @staticmethod
    def augment(tokens, p=0.15):
        kept = [t for t in tokens if random.random() >= p]
        if not kept:
            kept.append(random.choice(tokens))
        return kept

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self.augment(toks), self.augment(toks)


class SPRClassifierDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def encode_batch(token_lists):
    idx_seqs = [
        torch.tensor([vocab[t] for t in toks], dtype=torch.long) for toks in token_lists
    ]
    lengths = torch.tensor([len(s) for s in idx_seqs], dtype=torch.long)
    padded = pad_sequence(idx_seqs, batch_first=True, padding_value=pad_id)
    return padded.to(device), lengths.to(device)


def collate_contrastive(batch):
    v1, v2 = zip(*batch)
    ids1, len1 = encode_batch(v1)
    ids2, len2 = encode_batch(v2)
    return {"ids1": ids1, "len1": len1, "ids2": ids2, "len2": len2}


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids, lens = encode_batch(toks)
    return {
        "ids": ids,
        "len": lens,
        "label": torch.tensor(labels, device=device),
        "sequence": raw,
    }


pretrain_loader = DataLoader(
    SPRContrastiveDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)


# -------------------------------------------------
# Model --------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        emb = self.emb(ids)
        packed = pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder: Encoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# -------------------------------------------------
# Contrastive loss (NT_Xent)
def nt_xent_loss(z1, z2, temp=0.5):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(N, 2 * N, device=device)
    pos_sim = torch.cat([sim[:N, N:].diag(), sim[N:, :N].diag()])
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


# -------------------------------------------------
# Experiment tracking dict ------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "ACS": [], "SCHM": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}

# -------------------------------------------------
# Pre-training ----------------------------------------------------------
encoder = Encoder(vocab_size).to(device)
optim_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pre_epochs = 2
for ep in range(1, pre_epochs + 1):
    encoder.train()
    running = 0.0
    for batch in pretrain_loader:
        optim_pt.zero_grad()
        z1 = encoder(batch["ids1"], batch["len1"])
        z2 = encoder(batch["ids2"], batch["len2"])
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optim_pt.step()
        running += loss.item() * batch["ids1"].size(0)
    print(
        f"Pre-train epoch {ep}/{pre_epochs}: loss={running/len(pretrain_loader.dataset):.4f}"
    )

base_state = copy.deepcopy(encoder.state_dict())

# -------------------------------------------------
# Fine-tuning -----------------------------------------------------------
num_classes = len(set(spr_bench["train"]["label"]))
criterion = nn.CrossEntropyLoss()
epochs = 20
patience, wait, best_val = 3, 0, float("inf")

encoder = Encoder(vocab_size).to(device)
encoder.load_state_dict(base_state)
clf = Classifier(encoder, num_classes=num_classes).to(device)
optim_ft = torch.optim.Adam(clf.parameters(), lr=0.003)  # tuned lr


def compute_acs(model, sequences, K=3):
    model.eval()
    consistent = 0
    total = len(sequences)
    with torch.no_grad():
        for seq in sequences:
            toks = tokenize(seq)
            ids, lens = encode_batch([toks])
            base_pred = model(ids, lens).argmax(1)
            ok = True
            for _ in range(K):
                aug = SPRContrastiveDataset.augment(toks)
                ids_a, lens_a = encode_batch([aug])
                if model(ids_a, lens_a).argmax(1) != base_pred:
                    ok = False
                    break
            consistent += int(ok)
    return consistent / total if total else 0.0


for ep in range(1, epochs + 1):
    clf.train()
    train_loss = 0.0
    for batch in train_loader:
        optim_ft.zero_grad()
        logits = clf(batch["ids"], batch["len"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        optim_ft.step()
        train_loss += loss.item() * batch["ids"].size(0)
    train_loss /= len(train_loader.dataset)

    # -------- validation ---------
    clf.eval()
    val_loss, preds, trues, seqs = 0.0, [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            logits = clf(batch["ids"], batch["len"])
            loss = criterion(logits, batch["label"])
            val_loss += loss.item() * batch["ids"].size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(batch["label"].cpu().tolist())
            seqs.extend(batch["sequence"])
    val_loss /= len(dev_loader.dataset)

    swa = shape_weighted_accuracy(seqs, trues, preds)
    cwa = color_weighted_accuracy(seqs, trues, preds)
    acs = compute_acs(clf, seqs, K=3)
    schm = 2 * swa * cwa / (swa + cwa) if swa + cwa else 0.0

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
    experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
    experiment_data["SPR_BENCH"]["metrics"]["ACS"].append(acs)
    experiment_data["SPR_BENCH"]["metrics"]["SCHM"].append(schm)
    experiment_data["SPR_BENCH"]["epochs"].append(ep)

    print(
        f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"SWA={swa:.3f} CWA={cwa:.3f} ACS={acs:.3f} SCHM={schm:.3f}"
    )

    # early stopping
    if val_loss < best_val - 1e-4:
        best_val, wait = val_loss, 0
        experiment_data["SPR_BENCH"]["predictions"] = preds.copy()
        experiment_data["SPR_BENCH"]["ground_truth"] = trues.copy()
        torch.save(clf.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# -------------------------------------------------
# Plot & Save ----------------------------------------------------------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["losses"]["train"], label="train")
plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"], label="val")
plt.title("Fine-tune Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training complete. Artifacts saved to ./working/")
