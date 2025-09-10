import os, time, random, pathlib, math, copy, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------
# mandatory working directory and device boiler-plate
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------
# helpers ------------------------------------------------------
def count_color_variety(seq):  # token = ShapeChar + ColorDigit
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    num = sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p)
    den = max(sum(w), 1)
    return num / den


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    num = sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p)
    den = max(sum(w), 1)
    return num / den


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p)
    den = max(sum(w), 1)
    return num / den


# -------------------------------------------------------------
# data utilities ----------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


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
    if root.exists():
        print("Using real SPR_BENCH")
        return load_spr_bench(root)
    else:
        print("SPR_BENCH not found – generating synthetic.")
        train = load_dataset("json", data_files=[make_synth(4000)], split="train")
        dev = load_dataset(
            "json", data_files=[make_synth(800, seed=999)], split="train"
        )
        test = load_dataset(
            "json", data_files=[make_synth(1000, seed=555)], split="train"
        )
        return DatasetDict(train=train, dev=dev, test=test)


dset = get_dataset()
num_classes = len(set(dset["train"]["label"]))
print("Num classes =", num_classes)

# -------------------------------------------------------------
# vocabulary & glyph clustering (still done, though clusters unused by model)
all_tokens = set(tok for seq in dset["train"]["sequence"] for tok in seq.split())
vocab = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 = pad
vocab_size = len(vocab) + 1
print("Vocab size =", vocab_size)

# simple autoencoder to learn token embeddings
onehots = np.eye(vocab_size - 1, dtype=np.float32)
ae_dim = 4
ae = nn.Sequential(
    nn.Linear(vocab_size - 1, ae_dim), nn.Tanh(), nn.Linear(ae_dim, vocab_size - 1)
).to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-2)
criterion_ae = nn.MSELoss()
onehots_t = torch.tensor(onehots).to(device)
for epoch in range(120):  # shorter, adequate
    opt_ae.zero_grad()
    recon = ae(onehots_t)
    loss = criterion_ae(recon, onehots_t)
    loss.backward()
    opt_ae.step()
    if epoch % 40 == 0:
        print(f"AE epoch {epoch} loss {loss.item():.4f}")
with torch.no_grad():
    latents = ae[:2](onehots_t).cpu().numpy()

K = 8
km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(latents)
cluster_ids = km.labels_
cluster_map = {tok: cluster_ids[i] for i, tok in enumerate(sorted(all_tokens))}
print("Cluster counts:", np.bincount(cluster_ids))


# -------------------------------------------------------------
# torch Dataset / Dataloader -----------------------------------
def encode_seq(seq):
    ids = [vocab.get(tok, 0) for tok in seq.split()]
    clus = [cluster_map.get(tok, 0) for tok in seq.split()]
    return ids, clus


class SPRTorch(Dataset):
    def __init__(self, split):
        enc = [encode_seq(s) for s in split["sequence"]]
        self.ids = [e[0] for e in enc]
        self.clu = [e[1] for e in enc]
        self.labels = split["label"]
        self.raw = split["sequence"]

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
    maxlen = max(len(b["tok"]) for b in batch)
    B = len(batch)
    tok_pad = torch.zeros((B, maxlen), dtype=torch.long)
    clu_pad = torch.zeros((B, maxlen), dtype=torch.long)
    lens, labels, raws = [], [], []
    for i, b in enumerate(batch):
        L = len(b["tok"])
        tok_pad[i, :L] = b["tok"]
        clu_pad[i, :L] = b["clu"]
        lens.append(L)
        labels.append(b["label"])
        raws.append(b["raw"])
    return {
        "tok": tok_pad,
        "clu": clu_pad,
        "len": torch.tensor(lens),
        "label": torch.stack(labels),
        "raw": raws,
    }


train_ds, dev_ds = SPRTorch(dset["train"]), SPRTorch(dset["dev"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
train_eval_loader = DataLoader(
    train_ds, batch_size=128, shuffle=False, collate_fn=collate
)  # no shuffle!
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate)


# -------------------------------------------------------------
# model --------------------------------------------------------
class NoClusterClassifier(nn.Module):
    def __init__(self, vocab_sz, emb=32, hid=64, classes=3):
        super().__init__()
        self.emb_tok = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hid * 2, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, tok, len_):
        x = self.emb_tok(tok)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, len_.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)


model = NoClusterClassifier(vocab_size, classes=num_classes).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------------------
# experiment data container -----------------------------------
experiment_data = {
    "remove_cluster_embedding": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        }
    }
}
ed = experiment_data["remove_cluster_embedding"]["SPR_BENCH"]


# -------------------------------------------------------------
# evaluation loop (order-safe) ---------------------------------
def eval_loader(loader):
    model.eval()
    all_preds, all_labels, all_raws = [], [], []
    with torch.no_grad():
        for batch in loader:
            # move tensors to device
            batch_t = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch_t["tok"], batch_t["len"])
            preds = logits.argmax(1).cpu().tolist()
            labels = batch_t["label"].cpu().tolist()
            raws = batch["raw"]  # list of strings, already in correct order
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_raws.extend(raws)
    return all_preds, all_labels, all_raws


# -------------------------------------------------------------
# training -----------------------------------------------------
MAX_EPOCHS, patience = 25, 5
best_val_pcwa, stall = -1, 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    # ------ train ------------------------------------------------
    model.train()
    running_loss, running_n = 0.0, 0
    for batch in train_loader:
        batch_t = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch_t["tok"], batch_t["len"])
        loss = criterion_cls(logits, batch_t["label"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_t["label"].size(0)
        running_n += batch_t["label"].size(0)
    train_loss = running_loss / running_n

    # ------ evaluate -------------------------------------------
    train_preds, train_labels, train_raws = eval_loader(train_eval_loader)
    val_preds, val_labels, val_raws = eval_loader(dev_loader)

    tr_pcwa = pcwa(train_raws, train_labels, train_preds)
    v_pcwa = pcwa(val_raws, val_labels, val_preds)
    v_cwa, v_swa = cwa(val_raws, val_labels, val_preds), swa(
        val_raws, val_labels, val_preds
    )

    # record
    ed["epochs"].append(epoch)
    ed["losses"]["train"].append(train_loss)
    ed["metrics"]["train"].append({"pcwa": tr_pcwa})
    ed["metrics"]["val"].append({"pcwa": v_pcwa, "cwa": v_cwa, "swa": v_swa})

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  "
        f"val_pcwa={v_pcwa:.4f} (CWA {v_cwa:.3f} SWA {v_swa:.3f}) "
        f"time {time.time()-t0:.1f}s"
    )

    # early stopping on PCWA
    if v_pcwa > best_val_pcwa + 1e-6:
        best_val_pcwa = v_pcwa
        stall = 0
        ed["predictions"] = val_preds
        ed["ground_truth"] = val_labels
    else:
        stall += 1
        if stall >= patience:
            print("Early stopping.")
            break

# -------------------------------------------------------------
# save & plot --------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

val_pcwa = [m["pcwa"] for m in ed["metrics"]["val"]]
plt.figure()
plt.plot(ed["epochs"], val_pcwa, marker="o")
plt.title("Val PCWA (NoCluster)")
plt.xlabel("Epoch")
plt.ylabel("PCWA")
plt.savefig(os.path.join(working_dir, "val_pcwa_nocluster.png"))
print("Training complete – artefacts saved in ./working")
