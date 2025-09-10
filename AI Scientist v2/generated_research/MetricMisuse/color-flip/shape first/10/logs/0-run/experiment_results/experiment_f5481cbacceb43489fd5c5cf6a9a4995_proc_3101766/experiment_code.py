import os, random, math, time, pathlib, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset as HFDataset

# ----------------------------------------------------------------------
# working directory & experiment tracking
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

# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------- SPR helper functions (copied from spec) -------------
def count_shape_variety(sequence: str) -> int:
    return len(set(token[0] for token in sequence.strip().split() if token))


def count_color_variety(sequence: str) -> int:
    return len(set(token[1] for token in sequence.strip().split() if len(token) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ----------------------------------------------------------------------
#  DATA  ----------------------------------------------------------------
SHAPES = list("ABCDE")
COLORS = list("rgbyp")


def make_random_sequence(min_len=4, max_len=12):
    length = random.randint(min_len, max_len)
    tokens = [random.choice(SHAPES) + random.choice(COLORS) for _ in range(length)]
    return " ".join(tokens)


def synthetic_dataset(n):
    seqs, labels = [], []
    for _ in range(n):
        s = make_random_sequence()
        # label = 1 if even #unique shapes else 0
        label = 1 if count_shape_variety(s) % 2 == 0 else 0
        seqs.append(s)
        labels.append(label)
    return {"sequence": seqs, "label": labels}


def load_spr_bench_or_synth():
    data_root = os.environ.get("SPR_DIR", "./SPR_BENCH")
    root = pathlib.Path(data_root)
    if root.exists():
        print(f"Found SPR_BENCH at {root}")

        # small inline loader so we don't rely on external file
        def _ld(csv):
            return HFDataset.from_csv(root / csv)

        d = DatasetDict(
            {"train": _ld("train.csv"), "dev": _ld("dev.csv"), "test": _ld("test.csv")}
        )
    else:
        print("SPR_BENCH not found – creating synthetic toy data.")
        d = DatasetDict(
            {
                "train": HFDataset.from_dict(synthetic_dataset(2000)),
                "dev": HFDataset.from_dict(synthetic_dataset(400)),
                "test": HFDataset.from_dict(synthetic_dataset(400)),
            }
        )
    return d


dsets = load_spr_bench_or_synth()

# ---------------- Tokeniser -------------------------------------------
PAD_ID = 0


def build_vocab(dataset):
    vocab = set()
    for s in dataset["train"]["sequence"]:
        vocab.update(s.split())
    itos = ["<PAD>"] + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


stoi, itos = build_vocab(dsets)
VOCAB_SIZE = len(itos)
MAX_LEN = max(len(s.split()) for s in dsets["train"]["sequence"])  # keep simple


def encode(seq):
    ids = [stoi[tok] for tok in seq.split()]
    if len(ids) < MAX_LEN:
        ids += [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids


for split in dsets.keys():
    dsets[split] = dsets[split].map(
        lambda ex: {"input_ids": encode(ex["sequence"])}, remove_columns=[]
    )


# ---------------- PyTorch Dataset -------------------------------------
class SPRTorchSet(Dataset):
    def __init__(self, hf_dataset):
        self.ids = hf_dataset["input_ids"]
        self.seqs = hf_dataset["sequence"]
        self.labels = hf_dataset["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "sequence": self.seqs[idx],
        }


batch_size = 128
train_loader = DataLoader(
    SPRTorchSet(dsets["train"]), batch_size=batch_size, shuffle=True
)
dev_loader = DataLoader(SPRTorchSet(dsets["dev"]), batch_size=batch_size)
test_loader = DataLoader(SPRTorchSet(dsets["test"]), batch_size=batch_size)


# ----------------- Augmentations --------------------------------------
def augment_sequence(tokens):
    tokens = tokens.copy()
    # masking 15%
    for i in range(len(tokens)):
        if random.random() < 0.15:
            tokens[i] = "<MASK>"
    # 20% shuffle (swap)
    if len(tokens) > 2 and random.random() < 0.2:
        i, j = random.sample(range(len(tokens)), 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


def augment_batch(batch_ids):
    aug = []
    for ids in batch_ids:
        toks = [itos[i] for i in ids if i != PAD_ID][:]  # remove PAD
        toks_aug = augment_sequence(toks)
        ids_aug = [stoi.get(t, PAD_ID) for t in toks_aug]
        if len(ids_aug) < MAX_LEN:
            ids_aug += [PAD_ID] * (MAX_LEN - len(ids_aug))
        aug.append(ids_aug)
    return torch.tensor(aug, dtype=torch.long)


# ----------------- Model ----------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, ids):
        x = self.emb(ids)
        _, h = self.gru(x)
        return h.squeeze(0)  # [B,dim]


class SPRModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.enc = Encoder(vocab)
        self.head = nn.Linear(64, 2)

    def forward(self, ids):
        z = self.enc(ids)
        logits = self.head(z)
        return logits


model = SPRModel(VOCAB_SIZE).to(device)

# -------------- Contrastive loss --------------------------------------
temperature = 0.07


def info_nce(za, zb):
    B = za.size(0)
    z = torch.cat([za, zb], dim=0)  # 2B x d
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # 2B x 2B
    labels = torch.arange(B, device=device)
    labels = torch.cat([labels + B, labels], dim=0)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -9e15)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


# --------------- Training routines ------------------------------------
ce_loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_pretrain(epochs=5):
    model.train()
    for epoch in range(1, epochs + 1):
        tot_loss = 0
        nb = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            ids_aug = augment_batch(ids.cpu().tolist()).to(device)
            opt.zero_grad()
            za = model.enc(ids)
            zb = model.enc(ids_aug)
            loss = info_nce(za, zb)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        print(f"Pretrain epoch {epoch}: contrastive_loss = {tot_loss/nb:.4f}")


def evaluate(loader):
    model.eval()
    preds, gts, seqs = [], [], []
    tot_loss = 0
    nb = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids)
            loss = ce_loss(logits, labels)
            tot_loss += loss.item()
            nb += 1
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p)
            gts.extend(labels.cpu().tolist())
            seqs.extend(batch["sequence"])
    swa = shape_weighted_accuracy(seqs, gts, preds)
    cwa = color_weighted_accuracy(seqs, gts, preds)
    return tot_loss / nb, preds, gts, seqs, swa, cwa


def compute_acs(loader, k_aug=3):
    model.eval()
    consistent = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["label"].tolist()
            base_pred = model(ids).argmax(1).cpu().tolist()
            for i in range(ids.size(0)):
                good = True
                if base_pred[i] != labels[i]:
                    good = False
                else:
                    for _ in range(k_aug):
                        aug = augment_batch([ids[i].cpu().tolist()]).to(device)
                        p = model(aug).argmax(1).item()
                        if p != base_pred[i]:
                            good = False
                            break
                if good:
                    consistent += 1
                total += 1
    return consistent / total if total else 0.0


# -------------------- MAIN EXPERIMENT ---------------------------------
start = time.time()
run_pretrain()

epochs_sup = 5
for epoch in range(1, epochs_sup + 1):
    model.train()
    tot_train = 0
    nb = 0
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        opt.zero_grad()
        logits = model(ids)
        loss = ce_loss(logits, labels)
        loss.backward()
        opt.step()
        tot_train += loss.item()
        nb += 1
    train_loss = tot_train / nb

    val_loss, preds, gts, seqs, swa, cwa = evaluate(dev_loader)
    acs = compute_acs(dev_loader)
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} ACS={acs:.3f}"
    )

    experiment_data["SPR"]["losses"]["train"].append(train_loss)
    experiment_data["SPR"]["losses"]["val"].append(val_loss)
    experiment_data["SPR"]["metrics"]["val"].append(
        {"swa": swa, "cwa": cwa, "acs": acs}
    )
    experiment_data["SPR"]["metrics"]["train"].append({"loss": train_loss})

# -------------------- TEST EVALUATION ---------------------------------
test_loss, preds, gts, seqs, swa_t, cwa_t = evaluate(test_loader)
acs_t = compute_acs(test_loader)
print(f"TEST => loss:{test_loss:.4f} SWA:{swa_t:.3f} CWA:{cwa_t:.3f} ACS:{acs_t:.3f}")
experiment_data["SPR"]["predictions"] = preds
experiment_data["SPR"]["ground_truth"] = gts
experiment_data["SPR"]["acs"].append(acs_t)

# -------------------- SAVE METRICS ------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# -------------------- t-SNE Visual ------------------------------------
model.eval()
embs = []
lbls = []
with torch.no_grad():
    for batch in dev_loader:
        ids = batch["input_ids"].to(device)
        z = model.enc(ids).cpu().numpy()
        embs.append(z)
        lbls.extend(batch["label"].tolist())
embs = np.concatenate(embs, 0)
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
    embs
)
plt.figure(figsize=(6, 5))
plt.scatter(tsne[:, 0], tsne[:, 1], c=lbls, cmap="coolwarm", s=10)
plt.title("t-SNE of Dev Embeddings")
plt.savefig(os.path.join(working_dir, "tsne_dev_embeddings.png"), dpi=150)
plt.close()

print("Finished in %.1f s" % (time.time() - start))
