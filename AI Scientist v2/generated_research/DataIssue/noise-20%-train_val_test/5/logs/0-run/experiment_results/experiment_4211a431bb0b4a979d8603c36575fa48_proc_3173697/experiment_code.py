# MultiSyntheticDatasets ablation – single-file script
import os, random, math, pathlib, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

# ---------------------- GLOBAL CONFIG ------------------------------ #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"MultiSyntheticDatasets": {}}  # root container


# ------------------------ DATA BUILDERS ---------------------------- #
def _symbols(vocab_sz=12):
    return [chr(ord("A") + i) for i in range(vocab_sz)]


def build_even_parity(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    syms = _symbols(vocab_sz)

    def _split(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(label)
        return Dataset.from_dict(d)

    return DatasetDict(
        train=_split(num_train), dev=_split(num_dev), test=_split(num_test)
    )


def build_majority_symbol(
    num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12
):
    syms = _symbols(vocab_sz)

    def _split(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(syms) for _ in range(seqlen)]
            counts = {s: seq.count(s) for s in syms}
            max_ct = max(counts.values())
            majors = [s for s, c in counts.items() if c == max_ct]
            label_sym = random.choice(majors)
            label = syms.index(label_sym)
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(label)
        return Dataset.from_dict(d)

    return DatasetDict(
        train=_split(num_train), dev=_split(num_dev), test=_split(num_test)
    )


def is_cyclic_rotation(seq, pattern):
    """seq, pattern are lists of tokens"""
    if len(seq) != len(pattern):
        return False
    joined = pattern * 2
    for i in range(len(pattern)):
        if joined[i : i + len(pattern)] == seq:
            return True
    return False


def build_cyclic_shift(
    num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12
):
    syms = _symbols(vocab_sz)
    base_pattern = syms[:seqlen]  # "A B C ... "

    def rotate(lst, k):
        k %= len(lst)
        return lst[k:] + lst[:k]

    def _split(n):
        d = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            label = random.randint(0, 1)
            if label == 1:
                shift = random.randint(0, seqlen - 1)
                seq = rotate(base_pattern, shift)
            else:
                while True:
                    seq = [random.choice(syms) for _ in range(seqlen)]
                    if not is_cyclic_rotation(seq, base_pattern):
                        break
            d["id"].append(str(i))
            d["sequence"].append(" ".join(seq))
            d["label"].append(label)
        return Dataset.from_dict(d)

    return DatasetDict(
        train=_split(num_train), dev=_split(num_dev), test=_split(num_test)
    )


# --------------------------- VOCAB --------------------------------- #
def build_vocab(dataset: Dataset, field="sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for sent in dataset[field]:
        for tok in sent.split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    toks = [vocab.get(t, vocab["<unk>"]) for t in seq.split()]
    return toks[:max_len] if max_len else toks


# --------------------------- MODEL --------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_inv = (~mask).unsqueeze(-1).float()
        pooled = (h * mask_inv).sum(1) / mask_inv.sum(1).clamp(min=1)
        return self.cls(pooled)


# ---------------------- TRAIN / EVAL UTILS ------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    L = max(len(s) for s in seqs)
    pad_val = vocab["<pad>"]
    padded = [s + [pad_val] * (L - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_val
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            cnt += batch["labels"].size(0)
    return tot_loss / cnt, correct / cnt


# ------------------------- MAIN RUN -------------------------------- #
tasks = {
    "EvenParity": build_even_parity,
    "MajoritySymbol": build_majority_symbol,
    "CyclicShift": build_cyclic_shift,
}
nhead_values = [2, 4, 8, 16]
epochs = 5
embed_dim = 128
batch_size = 64

for task_name, builder in tasks.items():
    print(f"\n########## DATASET: {task_name} ##########")
    dsets = builder()
    vocab = build_vocab(dsets["train"])
    pad_idx = vocab["<pad>"]
    num_classes = len(set(dsets["train"]["label"]))
    print(f"Vocab={len(vocab)} | classes={num_classes}")
    loaders = {}
    for split in ["train", "dev", "test"]:
        loaders[split] = DataLoader(
            dsets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda b, v=vocab: collate_fn(b, v),
        )
    experiment_data["MultiSyntheticDatasets"][task_name] = {}
    for nhead in nhead_values:
        if embed_dim % nhead != 0:
            print(f"Skip nhead={nhead}")
            continue
        print(f"\n--- Training nhead={nhead} ---")
        model = SimpleTransformerClassifier(
            len(vocab), embed_dim, nhead, 2, num_classes, pad_idx
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        log = {"metrics": {"train": [], "val": []}, "losses": {"train": [], "val": []}}
        for epoch in range(1, epochs + 1):
            model.train()
            eloss = 0
            correct = 0
            tot = 0
            for batch in loaders["train"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = crit(logits, batch["labels"])
                loss.backward()
                opt.step()
                eloss += loss.item() * batch["labels"].size(0)
                correct += (logits.argmax(-1) == batch["labels"]).sum().item()
                tot += batch["labels"].size(0)
            tr_acc = correct / tot
            tr_loss = eloss / tot
            val_loss, val_acc = evaluate(model, loaders["dev"], crit)
            log["metrics"]["train"].append(tr_acc)
            log["metrics"]["val"].append(val_acc)
            log["losses"]["train"].append(tr_loss)
            log["losses"]["val"].append(val_loss)
            print(f"Ep{epoch}/{epochs} acc:{tr_acc:.3f}/{val_acc:.3f}")
        test_loss, test_acc = evaluate(model, loaders["test"], crit)
        preds, gts = [], []
        model.eval()
        with torch.no_grad():
            for batch in loaders["test"]:
                gbatch = {k: v.to(device) for k, v in batch.items()}
                logits = model(gbatch["input_ids"], gbatch["attention_mask"])
                preds.extend(logits.argmax(-1).cpu().tolist())
                gts.extend(batch["labels"].tolist())
        log["test_acc"] = test_acc
        log["predictions"] = preds
        log["ground_truth"] = gts
        experiment_data["MultiSyntheticDatasets"][task_name][str(nhead)] = log
        print(f"nhead={nhead} test_acc={test_acc:.3f}")

# ---------------------- SAVE RESULTS ------------------------------- #
save_path = os.path.join(working_dir, "experiment_data.npy")
np.save(save_path, experiment_data)
print("\nSaved experiment data to", save_path)
