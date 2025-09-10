import os, random, math, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset as HFDataset

# ------------------------------------------------------------- set-up
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


# ----------------------------------------------------- helper metrics
def count_shape_variety(seq):
    return len(set(t[0] for t in seq.split()))


def count_color_variety(seq):
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def shape_weighted_accuracy(seqs, y, yhat):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0


def color_weighted_accuracy(seqs, y, yhat):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if a == b else 0 for wi, a, b in zip(w, y, yhat)]
    return sum(c) / sum(w) if sum(w) else 0


# --------------------------------------------------------- data utils
SHAPES, COLORS = list("ABCDE"), list("rgbyp")


def make_random_sequence(min_len=4, max_len=12):
    return " ".join(
        random.choice(SHAPES) + random.choice(COLORS)
        for _ in range(random.randint(min_len, max_len))
    )


def synthetic_dataset(n):
    seqs, labels = [], []
    for _ in range(n):
        s = make_random_sequence()
        labels.append(int(count_shape_variety(s) % 2 == 0))
        seqs.append(s)
    return {"sequence": seqs, "label": labels}


def load_spr_bench_or_synth():
    root = pathlib.Path(os.environ.get("SPR_DIR", "./SPR_BENCH"))
    if root.exists():
        print("Found SPR_BENCH")

        def _ld(csv):
            return HFDataset.from_csv(root / csv)

        return DatasetDict(
            {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        )
    else:
        print("SPR_BENCH not found – using synthetic toy set.")
        return DatasetDict(
            {
                "train": HFDataset.from_dict(synthetic_dataset(3000)),
                "dev": HFDataset.from_dict(synthetic_dataset(600)),
                "test": HFDataset.from_dict(synthetic_dataset(600)),
            }
        )


dsets = load_spr_bench_or_synth()

# build vocab
PAD_ID = 0


def build_vocab(ds):
    vocab = set()
    for s in ds["train"]["sequence"]:
        vocab.update(s.split())
    itos = ["<PAD>"] + sorted(vocab) + ["<MASK>"]
    return {tok: i for i, tok in enumerate(itos)}, itos


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


class TorchSPR(Dataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        row = self.hf[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "sequence": row["sequence"],
        }


bs = 128
train_loader = DataLoader(TorchSPR(dsets["train"]), batch_size=bs, shuffle=True)
dev_loader = DataLoader(TorchSPR(dsets["dev"]), batch_size=bs)
test_loader = DataLoader(TorchSPR(dsets["test"]), batch_size=bs)

# -------------------------------------------------------- augmentation
MASK_ID = stoi["<MASK>"]


def augment_one(ids: list):
    ids = [x for x in ids if x != PAD_ID]
    # 1) random token mask
    ids = [MASK_ID if random.random() < 0.15 else t for t in ids]
    # 2) span shuffle (swap two spans)
    if len(ids) > 3 and random.random() < 0.2:
        i, j = sorted(random.sample(range(len(ids)), 2))
        ids[i], ids[j] = ids[j], ids[i]
    # 3) random drop (delete token)
    if len(ids) > 4 and random.random() < 0.1:
        ids.pop(random.randrange(len(ids)))
    ids = ids[:MAX_LEN] + [PAD_ID] * (MAX_LEN - len(ids))
    return ids


def augment_batch(batch_ids):
    return torch.tensor([augment_one(seq) for seq in batch_ids], dtype=torch.long)


# --------------------------------------------------------- Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(
        self, vocab, d_model=128, nhead=4, nlayers=2, dim_feed=256, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, MAX_LEN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feed, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, ids):
        x = self.emb(ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=(ids == PAD_ID))
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)  # [B,d]


class SPRNet(nn.Module):
    def __init__(self, vocab, feat_dim=128, proj_dim=64):
        super().__init__()
        self.enc = Encoder(vocab, d_model=feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        self.cls_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat_dim, 2))

    def forward(self, ids, proj=False):
        z = self.enc(ids)
        if proj:
            return nn.functional.normalize(self.proj(z), dim=1)
        return self.cls_head(z)


model = SPRNet(VOCAB_SIZE).to(device)

# ------------------------------------------------------- Training utils
temperature = 0.07


def info_nce(z1, z2):
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)  # 2B x d
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=device).bool()
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.arange(B, device=device)
    targets = torch.cat([targets + B, targets], 0)
    return nn.functional.cross_entropy(sim, targets)


ce_loss = nn.CrossEntropyLoss()


def run_pretrain(epochs=8):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for ep in range(1, epochs + 1):
        model.train()
        tot = nb = 0
        for batch in train_loader:
            ids = batch["input_ids"]
            view1 = ids
            view2 = augment_batch(ids.tolist())
            view1 = view1.to(device)
            view2 = view2.to(device)
            z1 = model(view1, proj=True)
            z2 = model(view2, proj=True)
            loss = info_nce(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"Pre-epoch {ep}: contrastive_loss = {tot/nb:.4f}")


run_pretrain()

# fresh optimiser for supervised phase
opt_cls = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=6)


def evaluate(loader):
    model.eval()
    preds = gts = seqs = []
    loss_tot = cnt = 0
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            lbl = b["label"].to(device)
            logits = model(ids)
            loss = ce_loss(logits, lbl)
            loss_tot += loss.item()
            cnt += 1
            p = logits.argmax(1).cpu().tolist()
            if not preds:
                preds = list(p)
            else:
                preds.extend(p)
            if not gts:
                gts = b["label"].tolist()
            else:
                gts.extend(b["label"].tolist())
            if not seqs:
                seqs = b["sequence"]
            else:
                seqs.extend(b["sequence"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    return loss_tot / cnt, swa, cwa, preds, gts, seqs


def compute_acs(loader, k_aug=3):
    model.eval()
    good = tot = 0
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"]
            lbl = b["label"]
            base = model(ids.to(device)).argmax(1).cpu()
            for i in range(ids.size(0)):
                consensus = True
                if base[i] != lbl[i]:
                    consensus = False
                else:
                    for _ in range(k_aug):
                        aug = augment_batch([ids[i].tolist()]).to(device)
                        if model(aug).argmax(1).item() != base[i]:
                            consensus = False
                            break
                if consensus:
                    good += 1
                tot += 1
    return good / tot if tot else 0.0


# ------------------------------------------------------ fine-tune
EPOCHS = 6
for ep in range(1, EPOCHS + 1):
    model.train()
    tot = cnt = 0
    for b in train_loader:
        ids = b["input_ids"].to(device)
        lbl = b["label"].to(device)
        logits = model(ids)
        loss = ce_loss(logits, lbl)
        opt_cls.zero_grad()
        loss.backward()
        opt_cls.step()
        tot += loss.item()
        cnt += 1
    train_loss = tot / cnt
    val_loss, swa, cwa, _, _, _ = evaluate(dev_loader)
    acs = compute_acs(dev_loader)
    print(
        f"Epoch {ep}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} ACS={acs:.3f}"
    )
    experiment_data["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "acs": acs}
    )
    experiment_data["SPR"]["metrics"]["train"].append({"loss": train_loss})
    scheduler.step()

# ------------------------------------------------------ test
test_loss, swa_t, cwa_t, preds, gts, seqs = evaluate(test_loader)
acs_t = compute_acs(test_loader)
print(f"TEST => loss:{test_loss:.4f} SWA:{swa_t:.3f} CWA:{cwa_t:.3f} ACS:{acs_t:.3f}")
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
experiment_data["SPR"]["acs"].append(acs_t)

# save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# t-SNE
model.eval()
embs = []
lbls = []
with torch.no_grad():
    for b in dev_loader:
        ids = b["input_ids"].to(device)
        embs.append(model.enc(ids).cpu().numpy())
        lbls.extend(b["label"])
embs = np.concatenate(embs, 0)
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
    embs
)
plt.scatter(tsne[:, 0], tsne[:, 1], c=lbls, cmap="Spectral", s=8)
plt.title("t-SNE Embedding (dev)")
plt.savefig(os.path.join(working_dir, "tsne_dev_embeddings.png"), dpi=140)
plt.close()
