import os, random, pathlib, math, itertools, time
from typing import List, Dict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset

# ------------------------------------------------------------
#  dirs / device ---------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
#  experiment container --------------------------------------
experiment_data = {
    "baseline_aug_aug": {},
    "one_sided_raw_aug": {},
}

# ------------------------------------------------------------
#  SPR loading (real or synthetic) ---------------------------
SHAPES, COLORS = list("ABCDEFGH"), list("01234567")


def generate_seq(min_len=5, max_len=15):
    L = random.randint(min_len, max_len)
    return " ".join(random.choice(SHAPES) + random.choice(COLORS) for _ in range(L))


def rule_label(seq: str):
    cnt = sum(tok[0] == "A" for tok in seq.split())
    return cnt % 4  # 4-class toy rule


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
        from SPR import load_spr_bench  # type: ignore
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


# ------------------------------------------------------------
#  Vocab/tokenisation ----------------------------------------
def build_vocab(dataset_split):
    vocab = {"<pad>": 0, "<mask>": 1}
    idx = 2
    for seq in dataset_split["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


vocab = build_vocab(spr["train"])
PAD, MASK = vocab["<pad>"], vocab["<mask>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq: str) -> List[int]:
    return [vocab[t] for t in seq.split()]


def augment(ids: List[int]) -> List[int]:
    # 50-50 choose mask or local shuffle
    if random.random() < 0.5:
        return [MASK if (i != PAD and random.random() < 0.15) else i for i in ids]
    # local shuffle
    ids = ids.copy()
    i, window = 0, 3
    while i < len(ids):
        j = min(len(ids), i + window)
        random.shuffle(ids[i:j])
        i += window
    return ids


# ------------------------------------------------------------
#  Datasets --------------------------------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, sequences, one_sided: bool = False):
        self.samples = [encode(s) for s in sequences]
        self.one_sided = one_sided

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        if self.one_sided:
            return torch.LongTensor(ids), torch.LongTensor(augment(ids))
        return torch.LongTensor(augment(ids)), torch.LongTensor(augment(ids))


class SupervisedDataset(Dataset):
    def __init__(self, sequences, labels):
        self.samples = [encode(s) for s in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx]), torch.tensor(self.labels[idx])


def pad_sequences(seq_list: List[torch.Tensor]) -> torch.Tensor:
    maxlen = max(len(s) for s in seq_list)
    return torch.stack(
        [torch.nn.functional.pad(s, (0, maxlen - len(s)), value=PAD) for s in seq_list]
    )


def collate_contrastive(batch):
    v1 = pad_sequences([b[0] for b in batch])
    v2 = pad_sequences([b[1] for b in batch])
    return {"v1": v1, "v2": v2}


def collate_supervised(batch):
    seqs = pad_sequences([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch]).long()
    return {"seq": seqs, "label": labels}


# ------------------------------------------------------------
#  Models ----------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.gru(emb)
        return torch.nn.functional.normalize(h.squeeze(0), dim=-1)


class SPRModel(nn.Module):
    def __init__(self, vocab_size, num_classes, hid=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, hid=hid)
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def nt_xent(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.t()) / temp
    sim.masked_fill_(torch.eye(2 * B, device=z.device).bool(), -9e15)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return nn.CrossEntropyLoss()(sim, pos)


# ------------------------------------------------------------
#  common helpers --------------------------------------------
num_classes = len(set(spr["train"]["label"]))
batch_size = 128
pre_epochs, sup_epochs = 2, 3


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, correct, preds_all, gts_all = 0, 0, [], []
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
        correct / len(loader.dataset),
        preds_all,
        gts_all,
    )


def augmentation_consistency(model, loader, variants=3):
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
                    aug_ids = torch.LongTensor(augment(ids))
                    aug_ids = torch.nn.functional.pad(
                        aug_ids, (0, s.size(0) - aug_ids.size(0)), value=PAD
                    ).unsqueeze(0)
                    aug_pred = model(aug_ids.to(device)).argmax().item()
                    if aug_pred != base:
                        ok = False
                total += 1
                consistent += ok
    return consistent / total


# ------------------------------------------------------------
#  core experiment runner ------------------------------------
def run_experiment(tag: str, one_sided: bool):
    print(f"\n=== Running experiment: {tag} | one_sided={one_sided} ===")
    # containers
    experiment_data[tag] = {
        "SPR": {
            "metrics": {"val_ACS": []},
            "losses": {"contrastive": [], "train_sup": [], "val_sup": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
    # loaders
    contrast_loader = DataLoader(
        ContrastiveDataset(spr["train"]["sequence"], one_sided=one_sided),
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
    # contrastive pre-training
    encoder = Encoder(vocab_size).to(device)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for ep in range(1, pre_epochs + 1):
        encoder.train()
        tot = 0
        for batch in contrast_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = nt_xent(encoder(batch["v1"]), encoder(batch["v2"]))
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            tot += loss.item() * batch["v1"].size(0)
        epoch_loss = tot / len(contrast_loader.dataset)
        experiment_data[tag]["SPR"]["losses"]["contrastive"].append(epoch_loss)
        print(f" contrastive ep {ep}: loss={epoch_loss:.4f}")
    # supervised fine-tune
    model = SPRModel(vocab_size, num_classes).to(device)
    model.encoder.load_state_dict(encoder.state_dict())
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, sup_epochs + 1):
        model.train()
        run_loss = 0
        for batch in sup_train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = criterion(model(batch["seq"]), batch["label"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * batch["seq"].size(0)
        train_loss = run_loss / len(sup_train_loader.dataset)
        val_loss, val_acc, preds, gts = evaluate(model, sup_val_loader, criterion)
        ACS = augmentation_consistency(model, sup_val_loader)
        # log
        ed = experiment_data[tag]["SPR"]
        ed["losses"]["train_sup"].append(train_loss)
        ed["losses"]["val_sup"].append(val_loss)
        ed["metrics"]["val_ACS"].append(ACS)
        ed["predictions"] = preds
        ed["ground_truth"] = gts
        ed["epochs"].append(ep)
        print(
            f"  finetune ep {ep}: train={train_loss:.4f} val={val_loss:.4f} "
            f"acc={val_acc:.4f} ACS={ACS:.4f}"
        )


# ------------------------------------------------------------
#  execute both variants -------------------------------------
run_experiment("baseline_aug_aug", one_sided=False)
run_experiment("one_sided_raw_aug", one_sided=True)

# ------------------------------------------------------------
#  save -------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nAll experiments finished. Data saved to 'working/experiment_data.npy'")
