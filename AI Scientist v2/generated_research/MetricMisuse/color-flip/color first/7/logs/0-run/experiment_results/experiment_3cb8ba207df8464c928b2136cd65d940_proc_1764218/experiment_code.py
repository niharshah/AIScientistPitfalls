import os, time, random, pathlib, math, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict
from collections import defaultdict

# ---------------------------------------------------------------------#
# 0)  House-keeping & device
# ---------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------#
# 1)  Metric helpers
# ---------------------------------------------------------------------#
def _count_color(seq):  # token = ShapeChar + ColorDigit
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def _count_shape(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_acc(seqs, y_true, y_pred):
    w = [_count_color(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def shape_weighted_acc(seqs, y_true, y_pred):
    w = [_count_shape(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def complexity_weighted_acc(seqs, y_true, y_pred):
    w = [_count_color(s) * _count_shape(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ---------------------------------------------------------------------#
# 2)  Dataset utilities
# ---------------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def make_synth(n, shapes=6, colors=5, max_len=8, num_labels=4, seed=123):
    rng = random.Random(seed)
    data = {"id": [], "sequence": [], "label": []}
    for i in range(n):
        L = rng.randint(3, max_len)
        seq = " ".join(
            chr(ord("A") + rng.randint(0, shapes - 1)) + str(rng.randint(0, colors - 1))
            for _ in range(L)
        )
        data["id"].append(str(i))
        data["sequence"].append(seq)
        data["label"].append(rng.randint(0, num_labels - 1))
    return data


def get_dataset():
    root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    try:
        if not root.exists():
            raise FileNotFoundError
        print("Using real SPR_BENCH")
        return load_spr_bench(root)
    except Exception:
        print("SPR_BENCH not found – generating synthetic.")
        train_json = make_synth(4000)
        dev_json = make_synth(800, seed=999)
        test_json = make_synth(1000, seed=555)
        train = load_dataset("json", data_files={"train": [train_json]}, split="train")
        dev = load_dataset("json", data_files={"train": [dev_json]}, split="train")
        test = load_dataset("json", data_files={"train": [test_json]}, split="train")
        return DatasetDict(train=train, dev=dev, test=test)


dset = get_dataset()
num_classes = len(set(dset["train"]["label"]))
print("Num classes =", num_classes)

# ---------------------------------------------------------------------#
# 3)  Vocab & glyph clustering (latent k-means)
# ---------------------------------------------------------------------#
all_tokens = set(tok for seq in dset["train"]["sequence"] for tok in seq.split())
vocab = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 = PAD
vocab_size = len(vocab) + 1
print("Vocab size =", vocab_size)

#  simple linear auto-encoder to get 4-D latent for each glyph
onehots = np.eye(vocab_size - 1, dtype=np.float32)
ae_dim = 4
ae = nn.Sequential(
    nn.Linear(vocab_size - 1, ae_dim), nn.Tanh(), nn.Linear(ae_dim, vocab_size - 1)
).to(device)
opt_ae, crit = torch.optim.Adam(ae.parameters(), lr=1e-2), nn.MSELoss()
onehots_t = torch.tensor(onehots, device=device)
for epoch in range(200):
    opt_ae.zero_grad()
    out = ae(onehots_t)
    loss = crit(out, onehots_t)
    loss.backward()
    opt_ae.step()
    if epoch % 80 == 0:
        print(f"AE epoch {epoch} loss {loss.item():.4f}")
with torch.no_grad():
    latents = ae[:2](onehots_t).cpu().numpy()

K = 8
km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(latents)
cluster_ids = km.labels_
cluster_map = {tok: cluster_ids[i] for i, tok in enumerate(sorted(all_tokens))}
print("Cluster counts:", np.bincount(cluster_ids))


# ---------------------------------------------------------------------#
# 4)  Torch dataset / dataloader
# ---------------------------------------------------------------------#
def encode_seq(seq):
    ids = [vocab.get(tok, 0) for tok in seq.split()]
    clu = [cluster_map.get(tok, 0) for tok in seq.split()]
    return ids, clu


class SPRTorch(Dataset):
    def __init__(self, hf_split):
        enc = [encode_seq(s) for s in hf_split["sequence"]]
        self.ids = [e[0] for e in enc]
        self.clu = [e[1] for e in enc]
        self.labels = hf_split["label"]
        self.raw = hf_split["sequence"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "tok": torch.tensor(self.ids[idx], dtype=torch.long),
            "clu": torch.tensor(self.clu[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": self.raw[idx],
        }


def collate(batch):
    B = len(batch)
    maxlen = max(len(b["tok"]) for b in batch)
    tok_pad = torch.zeros((B, maxlen), dtype=torch.long)
    clu_pad = torch.zeros((B, maxlen), dtype=torch.long)
    lens, labs, raws = [], [], []
    for i, b in enumerate(batch):
        L = len(b["tok"])
        tok_pad[i, :L] = b["tok"]
        clu_pad[i, :L] = b["clu"]
        lens.append(L)
        labs.append(b["label"])
        raws.append(b["raw"])
    return {
        "tok": tok_pad,
        "clu": clu_pad,
        "len": torch.tensor(lens, dtype=torch.long),
        "label": torch.stack(labs),
        "raw": raws,
    }


train_ds = SPRTorch(dset["train"])
dev_ds = SPRTorch(dset["dev"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
train_eval_loader = DataLoader(
    train_ds, batch_size=128, shuffle=False, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)


# ---------------------------------------------------------------------#
# 5)  Model (cluster info ignored in ablation)
# ---------------------------------------------------------------------#
class NoClusterClassifier(nn.Module):
    def __init__(self, vocab_sz, emb=32, hid=64, classes=2):
        super().__init__()
        self.emb_tok = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hid * 2, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, tok, clu, lens):  # `clu` unused
        x = self.emb_tok(tok)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


model = NoClusterClassifier(vocab_size, classes=num_classes).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # after model.to(device)

# ---------------------------------------------------------------------#
# 6)  Experiment data container
# ---------------------------------------------------------------------#
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
ed = experiment_data["SPR_BENCH"]


# ---------------------------------------------------------------------#
# 7)  Evaluation helpers
# ---------------------------------------------------------------------#
@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, labels, seqs = [], [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        # move tensors to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model(batch["tok"], batch["clu"], batch["len"])
        loss = criterion_cls(logits, batch["label"])
        total_loss += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        seqs.extend(batch["raw"])
    val_loss = total_loss / max(n, 1)
    cwa = color_weighted_acc(seqs, labels, preds)
    swa = shape_weighted_acc(seqs, labels, preds)
    pcwa = complexity_weighted_acc(seqs, labels, preds)  # Pattern-Complexity WA
    return val_loss, (cwa, swa, pcwa), preds, labels


# ---------------------------------------------------------------------#
# 8)  Training loop
# ---------------------------------------------------------------------#
MAX_EPOCHS, patience = 25, 5
best_pcwa, stall = -1.0, 0
for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss, m = 0.0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        out = model(batch["tok"], batch["clu"], batch["len"])
        loss = criterion_cls(out, batch["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)
        m += batch["label"].size(0)
    train_loss = running_loss / max(m, 1)

    # evaluate
    train_val_loss, (tr_cwa, tr_swa, tr_pcwa), _, _ = evaluate(train_eval_loader)
    dev_val_loss, (dv_cwa, dv_swa, dv_pcwa), val_preds, val_labels = evaluate(
        dev_loader
    )

    # store
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_val_loss)
    ed["losses"]["val"].append(dev_val_loss)
    ed["metrics"]["train"].append({"cwa": tr_cwa, "swa": tr_swa, "pcwa": tr_pcwa})
    ed["metrics"]["val"].append({"cwa": dv_cwa, "swa": dv_swa, "pcwa": dv_pcwa})

    print(
        f"Epoch {epoch:02d}: "
        f"val_loss = {dev_val_loss:.4f}  PCWA = {dv_pcwa:.4f} "
        f"(CWA {dv_cwa:.3f}  SWA {dv_swa:.3f})  time {time.time()-t0:.1f}s"
    )

    if dv_pcwa > best_pcwa + 1e-6:
        best_pcwa = dv_pcwa
        stall = 0
        ed["predictions"] = val_preds
        ed["ground_truth"] = val_labels
    else:
        stall += 1
        if stall >= patience:
            print("Early stopping.")
            break

# ---------------------------------------------------------------------#
# 9)  Save artefacts & plot
# ---------------------------------------------------------------------#
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

val_pcwa = [m["pcwa"] for m in ed["metrics"]["val"]]
plt.figure()
plt.plot(ed["epochs"], val_pcwa, marker="o")
plt.title("Validation Pattern-Complexity WA (NoCluster Ablation)")
plt.xlabel("Epoch")
plt.ylabel("PCWA")
plt.savefig(os.path.join(working_dir, "val_pcwa_nocluster.png"))
print("Training complete – artefacts saved to ./working")
