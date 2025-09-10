import os, random, pathlib, itertools, time, math
from typing import List, Dict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

# ---------------- working dir & device ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment container ----------------------
experiment_data = {
    "no_aug_contrast": {
        "SPR": {
            "metrics": {"train_ACS": [], "val_ACS": []},
            "losses": {"contrastive": [], "train_sup": [], "val_sup": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
ED = experiment_data["no_aug_contrast"]["SPR"]  # shortcut

# ---------------- synthetic / real SPR ----------------------
SHAPES, COLORS = list("ABCDEFGH"), list("01234567")


def generate_seq(min_len=5, max_len=15):
    return " ".join(
        random.choice(SHAPES) + random.choice(COLORS)
        for _ in range(random.randint(min_len, max_len))
    )


def rule_label(seq):
    return sum(tok[0] == "A" for tok in seq.split()) % 4


def synthetic_split(n):
    seqs, labels = [], []
    for i in range(n):
        s = generate_seq()
        seqs.append(s)
        labels.append(rule_label(s))
    return {"id": [str(i) for i in range(n)], "sequence": seqs, "label": labels}


def load_spr(split_counts=(4000, 1000, 1000)):
    root = pathlib.Path("SPR_BENCH")
    try:
        from SPR import load_spr_bench
    except ImportError:
        load_spr_bench = None
    if root.exists() and load_spr_bench is not None:
        print("Loading official SPR_BENCH dataset")
        return load_spr_bench(root)
    print("SPR_BENCH not found, building synthetic dataset")
    tr, dv, te = map(synthetic_split, split_counts)
    return DatasetDict(
        {
            "train": HFDataset.from_dict(tr),
            "dev": HFDataset.from_dict(dv),
            "test": HFDataset.from_dict(te),
        }
    )


spr = load_spr()


# ---------------- vocabulary & tokenization -----------------
def build_vocab(ds):
    vocab = {"<pad>": 0, "<mask>": 1}
    idx = 2
    for seq in ds["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(spr["train"])
PAD, MASK = vocab["<pad>"], vocab["<mask>"]
vocab_size = len(vocab)


def encode(seq):
    return [vocab[t] for t in seq.split()]


def decode(ids):
    inv = {i: t for t, i in vocab.items()}
    return [inv[i] for i in ids if i != PAD]


# ----------------- augmentations (kept for ACS only) --------
def token_mask(ids, prob=0.15):
    return [MASK if (i != PAD and random.random() < prob) else i for i in ids]


def local_shuffle(ids, window=3):
    ids = ids.copy()
    i = 0
    while i < len(ids):
        j = min(len(ids), i + window)
        random.shuffle(ids[i:j])
        i += window
    return ids


def augment(ids):
    return token_mask(ids) if random.random() < 0.5 else local_shuffle(ids)


# ---------------- PyTorch datasets --------------------------
class ContrastiveDatasetNoAug(Dataset):  # identical views
    def __init__(self, seqs):
        self.samples = [encode(s) for s in seqs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.LongTensor(self.samples[idx])
        return ids, ids  # no augmentation, identical


class SupervisedDataset(Dataset):
    def __init__(self, seqs, labels):
        self.samples = [encode(s) for s in seqs]
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx]), torch.tensor(self.labels[idx])


def pad_sequences(seq_list):
    maxlen = max(len(s) for s in seq_list)
    return torch.stack(
        [nn.functional.pad(s, (0, maxlen - len(s)), value=PAD) for s in seq_list]
    )


def collate_contrastive(batch):
    v1 = pad_sequences([b[0] for b in batch])
    v2 = pad_sequences([b[1] for b in batch])
    return {"v1": v1, "v2": v2}


def collate_supervised(batch):
    seqs = pad_sequences([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch]).long()
    return {"seq": seqs, "label": labels}


# ---------------- model definitions -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        h = h.squeeze(0)
        return nn.functional.normalize(h, dim=-1)


class SPRModel(nn.Module):
    def __init__(self, vocab_size, num_classes, hid=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, hid=hid)
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def nt_xent(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    sim.fill_diagonal_(-9e15)
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return nn.CrossEntropyLoss()(sim, pos_idx)


# ---------------- DataLoaders --------------------------------
batch_size = 128
contrast_loader = DataLoader(
    ContrastiveDatasetNoAug(spr["train"]["sequence"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_contrastive,
)
sup_train_loader = DataLoader(
    SupervisedDataset(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_supervised,
)
sup_val_loader = DataLoader(
    SupervisedDataset(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_supervised,
)

# ---------------- contrastive pre-training -------------------
encoder = Encoder(vocab_size).to(device)
enc_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
pre_epochs = 2
for epoch in range(1, pre_epochs + 1):
    encoder.train()
    total_loss = 0
    for batch in contrast_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = nt_xent(encoder(batch["v1"]), encoder(batch["v2"]))
        enc_opt.zero_grad()
        loss.backward()
        enc_opt.step()
        total_loss += loss.item() * batch["v1"].size(0)
    epoch_loss = total_loss / len(contrast_loader.dataset)
    ED["losses"]["contrastive"].append(epoch_loss)
    print(f"Contrastive epoch {epoch}: loss={epoch_loss:.4f}")

# ---------------- supervised fine-tuning ---------------------
num_classes = len(set(spr["train"]["label"]))
model = SPRModel(vocab_size, num_classes).to(device)
model.encoder.load_state_dict(encoder.state_dict())
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    tot_loss = correct = 0
    preds_all = []
    gts_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["seq"])
            loss = criterion(out, batch["label"])
            tot_loss += loss.item() * batch["seq"].size(0)
            preds = out.argmax(-1)
            correct += (preds == batch["label"]).sum().item()
            preds_all.extend(preds.cpu().tolist())
            gts_all.extend(batch["label"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        preds_all,
        gts_all,
        correct / len(loader.dataset),
    )


def augmentation_consistency(loader, variants=3):
    model.eval()
    total = consistent = 0
    with torch.no_grad():
        for batch in loader:
            seqs, labels = batch["seq"], batch["label"]
            for s, l in zip(seqs, labels):
                ids = [i.item() for i in s if i != PAD]
                base_pred = model(s.unsqueeze(0).to(device)).argmax().item()
                ok = base_pred == l.item()
                for _ in range(variants):
                    aug_ids = torch.LongTensor(augment(ids))
                    aug_ids = nn.functional.pad(
                        aug_ids, (0, s.size(0) - aug_ids.size(0)), value=PAD
                    ).unsqueeze(0)
                    if model(aug_ids.to(device)).argmax().item() != base_pred:
                        ok = False
                consistent += ok
                total += 1
    return consistent / total if total else 0.0


sup_epochs = 3
for epoch in range(1, sup_epochs + 1):
    model.train()
    run_loss = 0
    for batch in sup_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = criterion(model(batch["seq"]), batch["label"])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        run_loss += loss.item() * batch["seq"].size(0)
    train_loss = run_loss / len(sup_train_loader.dataset)
    val_loss, val_preds, val_gts, val_acc = evaluate(sup_val_loader)
    ACS = augmentation_consistency(sup_val_loader)
    ED["losses"]["train_sup"].append(train_loss)
    ED["losses"]["val_sup"].append(val_loss)
    ED["metrics"]["val_ACS"].append(ACS)
    ED["epochs"].append(epoch)
    ED["predictions"] = val_preds
    ED["ground_truth"] = val_gts
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} ACS={ACS:.4f}"
    )

# ---------------- save experiment data -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training complete, data saved to 'working/experiment_data.npy'")
