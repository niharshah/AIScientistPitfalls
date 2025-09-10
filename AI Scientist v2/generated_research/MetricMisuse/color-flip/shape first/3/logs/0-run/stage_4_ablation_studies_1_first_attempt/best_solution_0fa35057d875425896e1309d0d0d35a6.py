import os, random, pathlib, math, time, copy, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------#
# Boiler-plate & reproducibility
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# -----------------------------------------------------------------------------#
# Helper metrics
def count_shape_variety(sequence):  # first char of token
    return len(set(tok[0] for tok in sequence.strip().split()))


def count_color_variety(sequence):  # second char of token
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# -----------------------------------------------------------------------------#
# Synthetic dataset generators for three different rules
shapes, colors = list("ABCDE"), list("12345")


def gen_sequence():
    L = random.randint(3, 10)
    return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))


def label_rule1(seq):  # shape-vs-color variety
    return int(count_shape_variety(seq) >= count_color_variety(seq))


def label_rule2(seq):  # sequence length parity (even=1, odd=0)
    return int(len(seq.split()) % 2 == 0)


def label_rule3(seq):  # majority shape class exists
    toks = [t[0] for t in seq.split()]
    most = max(set(toks), key=toks.count)
    return int(toks.count(most) > len(toks) / 2)


RULES = {
    "variety": label_rule1,
    "length_parity": label_rule2,
    "majority_shape": label_rule3,
}


def make_dataset(n, rule_func):
    seqs, labels = [], []
    for _ in range(n):
        s = gen_sequence()
        seqs.append(s)
        labels.append(rule_func(s))
    return {"sequence": seqs, "label": labels}


splits = {}
for name, rule in RULES.items():
    splits[name] = {
        "train": make_dataset(6000, rule),
        "dev": make_dataset(1200, rule),
        "test": make_dataset(1200, rule),
    }

# -----------------------------------------------------------------------------#
# Shared vocabulary (union of all training sequences)
all_tokens = set()
for d in splits.values():
    for s in d["train"]["sequence"]:
        all_tokens.update(s.split())
vocab = {tok: i + 4 for i, tok in enumerate(sorted(all_tokens))}
vocab.update({"<pad>": 0, "<unk>": 1, "<mask>": 2, "<cls>": 3})
inv_vocab = {i: t for t, i in vocab.items()}
pad_id, unk_id, mask_id, cls_id = [
    vocab[t] for t in ["<pad>", "<unk>", "<mask>", "<cls>"]
]
max_len = (
    max(len(s.split()) for d in splits.values() for s in d["train"]["sequence"]) + 1
)


def encode(seq):
    ids = [cls_id] + [vocab.get(t, unk_id) for t in seq.split()]
    ids = ids[:max_len] + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# -----------------------------------------------------------------------------#
# Dataset wrappers
class SeqOnlyDS(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


class LabeledDS(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


# -----------------------------------------------------------------------------#
# Augmentation
def augment(seq: str):
    toks = seq.split()
    if random.random() < 0.5:
        toks = [t if random.random() > 0.3 else "<mask>" for t in toks]
    else:
        win = max(1, len(toks) // 4)
        i = random.randint(0, len(toks) - win)
        seg = toks[i : i + win]
        random.shuffle(seg)
        toks[i : i + win] = seg
    return " ".join(toks)


# -----------------------------------------------------------------------------#
# Model definitions
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.gru = nn.GRU(emb, hid, num_layers=layers, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return h[-1]  # (B,hid)


class Classifier(nn.Module):
    def __init__(self, encoder, hid, n_cls):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Linear(hid, n_cls)

    def forward(self, x):
        return self.fc(self.enc(x))


def contrastive_loss(z, temp=0.1):
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temp
    N = sim.size(0)
    sim.masked_fill_(torch.eye(N, device=z.device).bool(), -9e15)
    pos_idx = torch.arange(N, device=z.device) ^ 1
    return nn.functional.cross_entropy(sim, pos_idx)


# -----------------------------------------------------------------------------#
# Experiment data structure
experiment_data = {
    "multi_synth_generalization": {
        name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for name in RULES
    }
}
experiment_data["multi_synth_generalization"]["transfer"] = {}

# -----------------------------------------------------------------------------#
# Pre-training on union of unlabeled sequences
pretrain_seqs = []
for d in splits.values():
    pretrain_seqs.extend(d["train"]["sequence"])
pre_loader = DataLoader(
    SeqOnlyDS(pretrain_seqs), batch_size=256, shuffle=True, drop_last=True
)

encoder = Encoder(len(vocab)).to(device)
opt_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3)
print("\n--- Contrastive pre-training ---")
for ep in range(1, 9):
    encoder.train()
    tot = cnt = 0
    t0 = time.time()
    for batch in pre_loader:
        views = []
        for s in batch:
            views.append(encode(augment(s)))
            views.append(encode(augment(s)))
        x = torch.stack(views).to(device)
        opt_enc.zero_grad()
        loss = contrastive_loss(encoder(x))
        loss.backward()
        opt_enc.step()
        tot += loss.item()
        cnt += 1
    print(f"Epoch {ep}: loss={tot/cnt:.4f} ({time.time()-t0:.1f}s)")
pretrain_state = copy.deepcopy(encoder.state_dict())

# -----------------------------------------------------------------------------#
# Helper functions
crit = nn.CrossEntropyLoss()


def eval_model(model, loader):
    model.eval()
    loss_tot = n_tot = 0
    preds = []
    gts = []
    raws = []
    with torch.no_grad():
        for seqs, labels in loader:
            x = torch.stack([encode(s) for s in seqs]).to(device)
            y = torch.tensor(labels, device=device)
            logits = model(x)
            loss = crit(logits, y)
            loss_tot += loss.item() * y.size(0)
            n_tot += y.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            gts.extend(labels)
            raws.extend(seqs)
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    return loss_tot / n_tot, acc, preds, gts, raws


def aca_metric(model, seqs, labels, M=3):
    correct = total = 0
    with torch.no_grad():
        for s, l in zip(seqs, labels):
            variants = [s] + [augment(s) for _ in range(M)]
            xs = torch.stack([encode(v) for v in variants]).to(device)
            p = model(xs).argmax(1).cpu().tolist()
            correct += sum(int(pi == l) for pi in p)
            total += len(variants)
    return correct / total


# -----------------------------------------------------------------------------#
# Fine-tune on each dataset and evaluate cross-transfer
f_epochs = 10
for name in RULES:
    print(f"\n=== Fine-tuning on dataset: {name} ===")
    enc_i = Encoder(len(vocab)).to(device)
    enc_i.load_state_dict(pretrain_state)
    model = Classifier(enc_i, hid=256, n_cls=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        LabeledDS(splits[name]["train"]["sequence"], splits[name]["train"]["label"]),
        batch_size=128,
        shuffle=True,
    )
    dev_loader = DataLoader(
        LabeledDS(splits[name]["dev"]["sequence"], splits[name]["dev"]["label"]),
        batch_size=256,
    )
    for ep in range(1, f_epochs + 1):
        model.train()
        tot = n = 0
        for seqs, labels in train_loader:
            x = torch.stack([encode(s) for s in seqs]).to(device)
            y = torch.tensor(labels, device=device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = tot / n
        val_loss, val_acc, _, _, _ = eval_model(model, dev_loader)
        experiment_data["multi_synth_generalization"][name]["losses"]["train"].append(
            train_loss
        )
        experiment_data["multi_synth_generalization"][name]["losses"]["val"].append(
            val_loss
        )
        experiment_data["multi_synth_generalization"][name]["metrics"]["train"].append(
            train_loss
        )  # storing loss as proxy
        experiment_data["multi_synth_generalization"][name]["metrics"]["val"].append(
            val_acc
        )
        print(f"Epoch {ep}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    # Evaluation on own test set
    test_loader = DataLoader(
        LabeledDS(splits[name]["test"]["sequence"], splits[name]["test"]["label"]),
        batch_size=256,
    )
    tl, ta, tp, tg, tr = eval_model(model, test_loader)
    aca = aca_metric(
        model, splits[name]["test"]["sequence"], splits[name]["test"]["label"]
    )
    experiment_data["multi_synth_generalization"][name]["predictions"] = tp
    experiment_data["multi_synth_generalization"][name]["ground_truth"] = tg
    print(f"Test on {name}: loss={tl:.4f} | acc={ta:.4f} | ACA={aca:.4f}")

    # Cross-dataset transfer evaluation
    for other in RULES:
        if other == name:
            continue
        other_loader = DataLoader(
            LabeledDS(
                splits[other]["test"]["sequence"], splits[other]["test"]["label"]
            ),
            batch_size=256,
        )
        _, acc, pred, gt, _ = eval_model(model, other_loader)
        key = f"{name}_to_{other}"
        experiment_data["multi_synth_generalization"]["transfer"][key] = {
            "acc": acc,
            "predictions": pred,
            "ground_truth": gt,
        }
        print(f"  Transfer {name}->{other}: acc={acc:.4f}")

# -----------------------------------------------------------------------------#
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved results to working/experiment_data.npy")
