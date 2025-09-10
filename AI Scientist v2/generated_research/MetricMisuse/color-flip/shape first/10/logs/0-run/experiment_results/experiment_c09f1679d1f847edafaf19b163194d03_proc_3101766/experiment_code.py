import os, random, time, math, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset as HFDataset

# --------------------------------------------------------------------- #
#  house-keeping, experiment store
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "acs": [],
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------- #
#  helper metrics (spec)
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# --------------------------------------------------------------------- #
#  data loader (real or synthetic fallback)
SHAPES, COLORS = list("ABCDE"), list("rgbyp")


def make_random_seq(min_len=4, max_len=12):
    return " ".join(
        random.choice(SHAPES) + random.choice(COLORS)
        for _ in range(random.randint(min_len, max_len))
    )


def synth_set(n):
    seqs, labels = [], []
    for _ in range(n):
        s = make_random_seq()
        labels.append(1 if count_shape_variety(s) % 2 == 0 else 0)
        seqs.append(s)
    return {"sequence": seqs, "label": labels}


def load_spr():
    root = pathlib.Path(os.environ.get("SPR_DIR", "./SPR_BENCH"))
    if root.exists():
        print("Loading real SPR_BENCH")

        def _ld(csv):
            return HFDataset.from_csv(root / csv)

        return DatasetDict(
            {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        )
    print("Real data not found – using synthetic toy set")
    return DatasetDict(
        {
            "train": HFDataset.from_dict(synth_set(4000)),
            "dev": HFDataset.from_dict(synth_set(800)),
            "test": HFDataset.from_dict(synth_set(800)),
        }
    )


dsets = load_spr()

# --------------------------------------------------------------------- #
#  vocabulary + encoding
PAD_ID = 0
MASK_TOKEN = "<MASK>"


def build_vocab(ds):
    vocab = set()
    for s in ds["train"]["sequence"]:
        vocab.update(s.split())
    itos = ["<PAD>", MASK_TOKEN] + sorted(vocab)
    return {t: i for i, t in enumerate(itos)}, itos


stoi, itos = build_vocab(dsets)
VOCAB_SIZE = len(itos)
MAX_LEN = max(len(s.split()) for s in dsets["train"]["sequence"])


def encode(seq):
    ids = [stoi[t] for t in seq.split()]
    ids = ids[:MAX_LEN] + [PAD_ID] * (MAX_LEN - len(ids))
    return ids


for split in dsets:
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=[]
    )


class SPRTorch(Dataset):
    def __init__(self, hfset):
        self.ids = hfset["input_ids"]
        self.labels = hfset["label"]
        self.seq = hfset["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "sequence": self.seq[i],
        }


bs = 128
loaders = {
    split: DataLoader(SPRTorch(dsets[split]), batch_size=bs, shuffle=(split == "train"))
    for split in dsets
}


# --------------------------------------------------------------------- #
#  augmentations
def token_augment(tokens):
    t = tokens.copy()
    # mask 15%
    for i in range(len(t)):
        if random.random() < 0.15:
            t[i] = MASK_TOKEN
    # random swap 20%
    if len(t) > 2 and random.random() < 0.2:
        i, j = random.sample(range(len(t)), 2)
        t[i], t[j] = t[j], t[i]
    # colour/shape replacement 15%
    for i in range(len(t)):
        if random.random() < 0.15 and t[i] not in (MASK_TOKEN,):
            shape, colour = t[i][0], t[i][1]
            if random.random() < 0.5:
                shape = random.choice(SHAPES)
            else:
                colour = random.choice(COLORS)
            t[i] = shape + colour
    return t


def ids_from_tokens(tok_list):
    ids = [stoi.get(tok, stoi[MASK_TOKEN]) for tok in tok_list]
    return ids[:MAX_LEN] + [PAD_ID] * (MAX_LEN - len(ids))


def make_aug_views(batch_ids):
    view1, view2 = [], []
    for ids in batch_ids:
        toks = [itos[i] for i in ids if i != PAD_ID]
        view1.append(ids_from_tokens(token_augment(toks)))
        view2.append(ids_from_tokens(token_augment(toks)))
    return torch.tensor(view1), torch.tensor(view2)


# --------------------------------------------------------------------- #
#  model – transformer encoder + projection head
class SPRTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[:, : ids.size(1), :]
        mask = ids == PAD_ID
        z = self.encoder(x, src_key_padding_mask=mask)
        z = z.permute(0, 2, 1)
        z = self.pool(z).squeeze(-1)  # [B,d_model]
        return z


class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SPRTransformer(VOCAB_SIZE)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        self.cls = nn.Linear(128, 2)
        self.tau = nn.Parameter(torch.tensor(0.07))

    def forward(self, ids, proj=False):
        h = self.enc(ids)
        if proj:
            return nn.functional.normalize(self.proj(h), dim=1)
        return self.cls(h)


model = ContrastiveModel().to(device)

# --------------------------------------------------------------------- #
#  losses & optimisers
ce_loss = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)


def nt_xent(z1, z2, tau):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.mm(z, z.t()) / tau
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], 0)
    return nn.functional.cross_entropy(sim, targets)


# --------------------------------------------------------------------- #
#  Pre-training
def pretrain(epochs=5):
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0, 0
        for batch in loaders["train"]:
            ids = batch["input_ids"].to(device)
            v1, v2 = make_aug_views(ids.cpu().tolist())
            v1, v2 = v1.to(device), v2.to(device)
            z1 = model(v1, proj=True)
            z2 = model(v2, proj=True)
            loss = nt_xent(z1, z2, model.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        scheduler.step()
        print(f"Pre-E{ep}: contrastive_loss={tot/nb:.4f}")
        if tot / nb < 0.4:
            break


# --------------------------------------------------------------------- #
#  evaluation helpers
def evaluate(loader):
    model.eval()
    preds, gts, seqs = [], [], []
    tot, nb = 0, 0
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            lbl = b["label"].to(device)
            logit = model(ids)
            loss = ce_loss(logit, lbl)
            tot += loss.item()
            nb += 1
            preds.extend(logit.argmax(1).cpu().tolist())
            gts.extend(lbl.cpu().tolist())
            seqs.extend(b["sequence"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    return tot / nb, preds, gts, seqs, swa, cwa


def augmentation_consistency(loader, k=3):
    model.eval()
    good, tot = 0, 0
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            lbl = b["label"].tolist()
            base = model(ids).argmax(1).cpu().tolist()
            for i in range(len(base)):
                consistent = base[i] == lbl[i]
                if consistent:
                    for _ in range(k):
                        v, _ = make_aug_views([ids[i].cpu().tolist()])
                        p = model(v.to(device)).argmax(1).item()
                        if p != base[i]:
                            consistent = False
                            break
                good += int(consistent)
                tot += 1
    return good / tot if tot else 0.0


# --------------------------------------------------------------------- #
#  MAIN
start = time.time()
pretrain(epochs=6)

# supervised fine-tune
best_val = 1e9
patience = 3
wait = 0
for ep in range(1, 15):
    model.train()
    tot_t, nb = 0, 0
    for b in loaders["train"]:
        ids = b["input_ids"].to(device)
        lbl = b["label"].to(device)
        opt.zero_grad()
        loss = ce_loss(model(ids), lbl)
        loss.backward()
        opt.step()
        tot_t += loss.item()
        nb += 1
    train_loss = tot_t / nb
    val_loss, preds, gts, seqs, swa, cwa = evaluate(loaders["dev"])
    acs = augmentation_consistency(loaders["dev"])
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} ACS={acs:.3f}"
    )
    experiment_data["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "acs": acs}
    )
    experiment_data["SPR"]["metrics"]["train"].append({"loss": train_loss})
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        wait = 0
        best_state = model.state_dict()
    else:
        wait += 1
    if val_loss < 0.35 or wait >= patience:
        break
    scheduler.step()

model.load_state_dict(best_state)

# --------------------------------------------------------------------- #
#  TEST
test_loss, preds, gts, seqs, swa_t, cwa_t = evaluate(loaders["test"])
acs_t = augmentation_consistency(loaders["test"])
print(f"TEST => loss:{test_loss:.4f} SWA:{swa_t:.3f} CWA:{cwa_t:.3f} ACS:{acs_t:.3f}")
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
experiment_data["SPR"]["acs"].append(acs_t)

# save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# --------------------------------------------------------------------- #
#  t-SNE visualisation
model.eval()
embs, lbls = [], []
with torch.no_grad():
    for b in loaders["dev"]:
        ids = b["input_ids"].to(device)
        embs.append(model.enc(ids).cpu().numpy())
        lbls.extend(b["label"].tolist())
embs = np.concatenate(embs, 0)
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
    embs
)
plt.figure(figsize=(6, 5))
plt.scatter(tsne[:, 0], tsne[:, 1], c=lbls, cmap="coolwarm", s=8)
plt.title("t-SNE Dev Embeddings")
plt.savefig(os.path.join(working_dir, "tsne_dev_embeddings_transformer.png"), dpi=150)
plt.close()

print(f"Finished in {time.time()-start:.1f}s")
