import os, random, math, time, itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- working dir & GPU set-up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- synthetic SPR-style data ----------
random.seed(0)
shapes = list("ABCDE")  # 5 shapes
colors = list("123")  # 3 colours
pad_tok, mask_tok = "[PAD]", "[MASK]"
vocab = [pad_tok, mask_tok] + [s + c for s, c in itertools.product(shapes, colors)]
tok2id = {t: i for i, t in enumerate(vocab)}
id2tok = {i: t for t, i in tok2id.items()}
max_len = 12


def gen_seq():
    L = random.randint(6, 10)
    return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))


def label_rule(seq):
    s_var = len(set(t[0] for t in seq.split()))
    c_var = len(set(t[1] for t in seq.split()))
    return 1 if s_var > c_var else 0


def build_split(N):
    seqs = [gen_seq() for _ in range(N)]
    labels = [label_rule(s) for s in seqs]
    return seqs, labels


train_seqs, train_labels = build_split(2000)
dev_seqs, dev_labels = build_split(400)
test_seqs, test_labels = build_split(400)


# ---------- util metrics ----------
def pad_encode(seq):
    ids = [tok2id[tok] for tok in seq.split()]
    ids = ids[:max_len] + [tok2id[pad_tok]] * (max_len - len(ids))
    return ids


def count_shape_variety(sequence):
    return len(set(tok[0] for tok in sequence.split()))


def count_color_variety(sequence):
    return len(set(tok[1] for tok in sequence.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(correct) / sum(w)


# ---------- datasets ----------
def augment(seq):
    toks = seq.split()
    # token masking
    aug = [mask_tok if random.random() < 0.15 else t for t in toks]
    # local shuffle within window 2
    for i in range(len(aug) - 1):
        if random.random() < 0.3:
            aug[i], aug[i + 1] = aug[i + 1], aug[i]
    return " ".join(aug)


class ContrastiveDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        v1, v2 = augment(s), augment(s)
        return torch.tensor(pad_encode(v1)), torch.tensor(pad_encode(v2))


class ClassifyDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(pad_encode(self.seqs[idx])), torch.tensor(self.labels[idx])


# ---------- model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=tok2id[pad_tok])

    def forward(self, x):
        e = self.emb(x)  # B,L,D
        mask = (x != tok2id[pad_tok]).unsqueeze(-1)
        summed = (e * mask).sum(1)
        lens = mask.sum(1).clamp(min=1)
        return summed / lens  # B,D


class SPRModel(nn.Module):
    def __init__(self, enc, num_classes=2):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.head(self.enc(x))


# ---------- losses ----------
def simclr_loss(z1, z2, temp=0.07):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    B = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(representations, representations.T) / temp
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, -9e15)
    positives = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    numerator = torch.exp(positives)
    denominator = torch.exp(sim).sum(1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


# ---------- training helpers ----------
def compute_ACS(model, seqs, labels, n_aug=3):
    model.eval()
    with torch.no_grad():
        correct = 0
        for s, l in zip(seqs, labels):
            base = torch.tensor(pad_encode(s)).unsqueeze(0).to(device)
            pred0 = model(base).argmax(1).item()
            consistent = pred0 == l
            for _ in range(n_aug):
                aug_seq = augment(s)
                p = (
                    model(torch.tensor(pad_encode(aug_seq)).unsqueeze(0).to(device))
                    .argmax(1)
                    .item()
                )
                if p != pred0:
                    consistent = False
                    break
            correct += int(consistent)
    return correct / len(seqs)


experiment_data = {
    "spr_toy": {
        "metrics": {"val_loss": [], "ACS": [], "SWA": [], "CWA": []},
        "losses": {"pretrain": [], "finetune": []},
        "predictions": [],
        "ground_truth": test_labels,
    }
}

# ---------- pretraining ----------
enc = Encoder(len(vocab)).to(device)
optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
pretrain_loader = DataLoader(
    ContrastiveDataset(train_seqs), batch_size=128, shuffle=True
)

epochs_pt = 5
for ep in range(1, epochs_pt + 1):
    enc.train()
    losses = []
    for v1, v2 in pretrain_loader:
        v1, v2 = v1.to(device), v2.to(device)
        z1, z2 = enc(v1), enc(v2)
        loss = simclr_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    experiment_data["spr_toy"]["losses"]["pretrain"].append(mean_loss)
    print(f"Pretrain epoch {ep}: contrastive_loss = {mean_loss:.4f}")

# ---------- fine-tuning ----------
model = SPRModel(enc).to(device)
optim_f = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    ClassifyDataset(train_seqs, train_labels), batch_size=64, shuffle=True
)
dev_loader = DataLoader(ClassifyDataset(dev_seqs, dev_labels), batch_size=64)

epochs_ft = 6
for ep in range(1, epochs_ft + 1):
    model.train()
    tr_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optim_f.zero_grad()
        loss.backward()
        optim_f.step()
        tr_losses.append(loss.item())
    # validation
    model.eval()
    val_losses = []
    preds = []
    with torch.no_grad():
        for x, y in dev_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_losses.append(criterion(out, y).item())
            preds.extend(out.argmax(1).cpu().tolist())
    val_loss = np.mean(val_losses)
    experiment_data["spr_toy"]["losses"]["finetune"].append(np.mean(tr_losses))
    experiment_data["spr_toy"]["metrics"]["val_loss"].append(val_loss)
    # extra metrics
    swa = shape_weighted_accuracy(dev_seqs, dev_labels, preds)
    cwa = color_weighted_accuracy(dev_seqs, dev_labels, preds)
    acs = compute_ACS(model, dev_seqs, dev_labels)
    experiment_data["spr_toy"]["metrics"]["ACS"].append(acs)
    experiment_data["spr_toy"]["metrics"]["SWA"].append(swa)
    experiment_data["spr_toy"]["metrics"]["CWA"].append(cwa)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | ACS={acs:.3f} SWA={swa:.3f} CWA={cwa:.3f}"
    )

# ---------- test evaluation ----------
model.eval()
with torch.no_grad():
    test_pred = (
        model(torch.tensor([pad_encode(s) for s in test_seqs]).to(device))
        .argmax(1)
        .cpu()
        .tolist()
    )
experiment_data["spr_toy"]["predictions"] = test_pred
print("Test SWA:", shape_weighted_accuracy(test_seqs, test_labels, test_pred))
print("Test CWA:", color_weighted_accuracy(test_seqs, test_labels, test_pred))
print("Test ACS:", compute_ACS(model, test_seqs, test_labels))

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
