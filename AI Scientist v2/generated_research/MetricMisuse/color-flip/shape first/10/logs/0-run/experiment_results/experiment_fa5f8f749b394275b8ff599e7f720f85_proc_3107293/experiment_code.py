# ------------------------------------------------------------
#  Remove Contrastive Pre-training (Train Encoder From Scratch)
# ------------------------------------------------------------
import os, random, pathlib, math, itertools, time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import DatasetDict

# ------------------------------------ misc/paths/device -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------ experiment container ---
experiment_data = {
    "remove_contrastive_pretrain": {
        "SPR": {
            "metrics": {"val_acc": [], "val_ACS": []},
            "losses": {"train_sup": [], "val_sup": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}

# ------------------------------------ SPR loading ------------
SHAPES, COLORS = list("ABCDEFGH"), list("01234567")


def generate_seq(min_len=5, max_len=15):
    L = random.randint(min_len, max_len)
    return " ".join(random.choice(SHAPES) + random.choice(COLORS) for _ in range(L))


def rule_label(seq: str) -> int:
    return sum(tok[0] == "A" for tok in seq.split()) % 4


def synthetic_split(n: int):
    seqs, labels = [], []
    for i in range(n):
        s = generate_seq()
        seqs.append(s)
        labels.append(rule_label(s))
    return {"id": [str(i) for i in range(n)], "sequence": seqs, "label": labels}


def load_spr(split_counts=(4000, 1000, 1000)) -> DatasetDict:
    root = pathlib.Path("SPR_BENCH")
    try:
        from SPR import load_spr_bench  # type: ignore
    except ImportError:
        load_spr_bench = None
    if root.exists() and load_spr_bench:
        print("Loading official SPR_BENCH dataset")
        return load_spr_bench(root)
    print("SPR_BENCH not found, creating synthetic data")
    tr, dv, te = map(synthetic_split, split_counts)
    return DatasetDict(
        {
            "train": HFDataset.from_dict(tr),
            "dev": HFDataset.from_dict(dv),
            "test": HFDataset.from_dict(te),
        }
    )


spr = load_spr()


# ------------------------------------ vocabulary -------------
def build_vocab(ds_split):
    vocab, idx = {"<pad>": 0, "<mask>": 1}, 2
    for seq in ds_split["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(spr["train"])
PAD, MASK = vocab["<pad>"], vocab["<mask>"]
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


def augment(ids: List[int]) -> List[int]:
    # 50% token mask, else local shuffle inside windows of 3
    if random.random() < 0.5:
        return [MASK if (i != PAD and random.random() < 0.15) else i for i in ids]
    ids = ids.copy()
    i = 0
    while i < len(ids):
        j = min(len(ids), i + 3)
        random.shuffle(ids[i:j])
        i += 3
    return ids


# ------------------------------------ datasets ---------------
class SupervisedDataset(Dataset):
    def __init__(self, sequences, labels, training=False):
        self.samples = [encode(s) for s in sequences]
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        if self.training:  # optional augmentation during training
            ids = augment(ids) if random.random() < 0.5 else ids
        return torch.LongTensor(ids), torch.tensor(self.labels[idx]).long()


def pad_sequences(seq_list: List[torch.Tensor]) -> torch.Tensor:
    L = max(len(s) for s in seq_list)
    return torch.stack(
        [nn.functional.pad(s, (0, L - len(s)), value=PAD) for s in seq_list]
    )


def collate_sup(batch):
    seqs = pad_sequences([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return {"seq": seqs, "label": labels}


batch_size = 128
sup_train_loader = DataLoader(
    SupervisedDataset(spr["train"]["sequence"], spr["train"]["label"], training=True),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_sup,
)
sup_val_loader = DataLoader(
    SupervisedDataset(spr["dev"]["sequence"], spr["dev"]["label"], training=False),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_sup,
)


# ------------------------------------ model ------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)
        return h.squeeze(0)


class SPRModel(nn.Module):
    def __init__(self, vocab, num_classes, hid=128):
        super().__init__()
        self.encoder = Encoder(vocab, hid=hid)
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


num_classes = len(set(spr["train"]["label"]))
model = SPRModel(vocab_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------------ evaluation utils -------
def evaluate(loader):
    model.eval()
    tot_loss, correct, preds_all, gts_all = 0.0, 0, [], []
    with torch.no_grad():
        for batch in loader:
            seq, lbl = batch["seq"].to(device), batch["label"].to(device)
            out = model(seq)
            loss = criterion(out, lbl)
            tot_loss += loss.item() * seq.size(0)
            preds = out.argmax(-1)
            correct += (preds == lbl).sum().item()
            preds_all.extend(preds.cpu().tolist())
            gts_all.extend(lbl.cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        preds_all,
        gts_all,
        correct / len(loader.dataset),
    )


def augmentation_consistency(loader, variants=3):
    model.eval()
    total, consistent = 0, 0
    with torch.no_grad():
        for batch in loader:
            seqs, labels = batch["seq"], batch["label"]
            for s, l in zip(seqs, labels):
                ids = [i.item() for i in s if i != PAD]
                base = model(s.unsqueeze(0).to(device)).argmax().item()
                ok = base == l.item()
                for _ in range(variants):
                    aug = torch.LongTensor(augment(ids))
                    aug = nn.functional.pad(
                        aug, (0, s.size(0) - len(aug)), value=PAD
                    ).unsqueeze(0)
                    if model(aug.to(device)).argmax().item() != base:
                        ok = False
                total += 1
                consistent += ok
    return consistent / total if total else 0.0


# ------------------------------------ supervised training ----
sup_epochs = 3
for epoch in range(1, sup_epochs + 1):
    model.train()
    running = 0.0
    for batch in sup_train_loader:
        seq, lbl = batch["seq"].to(device), batch["label"].to(device)
        out = model(seq)
        loss = criterion(out, lbl)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += loss.item() * seq.size(0)

    train_loss = running / len(sup_train_loader.dataset)
    val_loss, preds, gts, val_acc = evaluate(sup_val_loader)
    val_ACS = augmentation_consistency(sup_val_loader)

    # log
    exp = experiment_data["remove_contrastive_pretrain"]["SPR"]
    exp["losses"]["train_sup"].append(train_loss)
    exp["losses"]["val_sup"].append(val_loss)
    exp["metrics"]["val_acc"].append(val_acc)
    exp["metrics"]["val_ACS"].append(val_ACS)
    exp["predictions"] = preds
    exp["ground_truth"] = gts
    exp["epochs"].append(epoch)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ACS={val_ACS:.4f}"
    )

# ------------------------------------ save -------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Data saved to working/experiment_data.npy")
