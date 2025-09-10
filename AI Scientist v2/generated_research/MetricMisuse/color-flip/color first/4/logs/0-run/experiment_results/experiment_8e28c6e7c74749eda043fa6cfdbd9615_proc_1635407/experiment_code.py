import os, pathlib, random, numpy as np, torch, warnings
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

# ---------- house-keeping & gpu ---------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment store ------------ #
experiment_data = {
    "default_spr": {
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
    }
}


# ---------- data helpers ---------------- #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv):  # helper
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def try_load_dataset() -> DatasetDict:
    default = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if default.exists():
        print("Loading real SPR_BENCH")
        return load_spr_bench(default)

    # ---- create tiny synthetic fallback ---- #
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def _gen(n):
        ids, seqs, labs = [], [], []
        for i in range(n):
            ids.append(str(i))
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            labs.append(random.choice(["ruleA", "ruleB", "ruleC"]))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labs})

    return DatasetDict(train=_gen(800), dev=_gen(160), test=_gen(160))


# ---------- metrics ---------------------- #
def _color_var(seq: str) -> int:
    # tolerate 1-char tokens (e.g. cluster ids)
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def _shape_var(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def weighted_acc(seqs, y_t, y_p, func):
    w = [func(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


def jwa(cwa, swa):
    # harmonic mean; avoid div0
    return 2 * cwa * swa / (cwa + swa) if (cwa + swa) > 0 else 0.0


# ---------- torch dataset --------------- #
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, lbl2id, token2id):
        self.seqs = hf_split["sequence"]
        self.raws = hf_split["sequence"]  # keep originals for metrics
        self.labels = [lbl2id[l] for l in hf_split["label"]]
        self.token2id = token2id

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [self.token2id[tok] for tok in self.seqs[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(self.labels[idx]),
            "raw": self.raws[idx],
        }


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    inp = torch.stack(
        [
            torch.nn.functional.pad(b["input_ids"], (0, maxlen - len(b["input_ids"])))
            for b in batch
        ]
    )
    lab = torch.stack([b["labels"] for b in batch])
    raw = [b["raw"] for b in batch]
    return {"input_ids": inp, "labels": lab, "raw": raw}


# ---------- simple encoder -------------- #
class EncoderClassifier(nn.Module):
    def __init__(self, vocab, embed=32, hid=64, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.rnn = nn.GRU(embed, hid, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(hid * 2, classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        h = torch.cat([h[0], h[1]], 1)
        return self.lin(h)


# ---------- training routine ------------ #
def train(max_epochs=5, lr=1e-3, batch=64):
    spr = try_load_dataset()

    # --- label & vocab --- #
    lbl2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    vocab = sorted({tok for seq in spr["train"]["sequence"] for tok in seq.split()})
    stoi = {tok: i + 1 for i, tok in enumerate(vocab)}  # +1 because 0 PAD
    num_cls = len(lbl2id)

    # --- dataloaders builder (token2id will mutate) --- #
    def mkloader(token2id, splits=("train", "dev")):
        loaders = []
        for sp in splits:
            ds = SPRTorchDataset(spr[sp], lbl2id, token2id)
            loaders.append(
                DataLoader(
                    ds,
                    batch_size=batch * (2 if sp == "dev" else 1),
                    shuffle=(sp == "train"),
                    collate_fn=collate,
                )
            )
        return loaders if len(loaders) > 1 else loaders[0]

    train_loader, dev_loader = mkloader(stoi)

    # --- model/optim --- #
    model = EncoderClassifier(len(stoi) + 1, classes=num_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    kmeans_done = False
    for ep in range(1, max_epochs + 1):
        # ----- train ----- #
        model.train()
        t_loss = 0.0
        for batch_data in train_loader:
            ids = batch_data["input_ids"].to(device)
            labels = batch_data["labels"].to(device)
            opt.zero_grad()
            out = model(ids)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            t_loss += loss.item() * labels.size(0)
        t_loss /= len(train_loader.dataset)
        experiment_data["default_spr"]["losses"]["train"].append((ep, t_loss))

        # ----- val  ------ #
        model.eval()
        v_loss, preds, gts, raws = 0.0, [], [], []
        with torch.no_grad():
            for batch_data in dev_loader:
                ids = batch_data["input_ids"].to(device)
                labels = batch_data["labels"].to(device)
                out = model(ids)
                loss = loss_fn(out, labels)
                v_loss += loss.item() * labels.size(0)
                p = torch.argmax(out, 1).cpu().tolist()
                preds += p
                gts += labels.cpu().tolist()
                raws += batch_data["raw"]
        v_loss /= len(dev_loader.dataset)
        acc = float(np.mean([p == l for p, l in zip(preds, gts)]))
        cwa = weighted_acc(raws, gts, preds, _color_var)
        swa = weighted_acc(raws, gts, preds, _shape_var)
        jwa_ = jwa(cwa, swa)
        experiment_data["default_spr"]["losses"]["val"].append((ep, v_loss))
        experiment_data["default_spr"]["metrics"]["val"].append(
            {"epoch": ep, "acc": acc, "cwa": cwa, "swa": swa, "jwa": jwa_}
        )
        print(
            f"Epoch {ep}: val_loss={v_loss:.4f} acc={acc:.3f} "
            f"CWA={cwa:.3f} SWA={swa:.3f} JWA={jwa_:.3f}"
        )

        # ----- latent clustering after 1st epoch ------ #
        if not kmeans_done and ep == 1:
            print(">>> Performing k-means clustering on embeddings …")
            with torch.no_grad():
                emb = model.embedding.weight.detach().cpu().numpy()[1:]  # skip PAD
            k = min(16, emb.shape[0])
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
            tok2cl = {tok: (cid + 1) for tok, cid in zip(vocab, km.labels_)}
            # replace embedding matrix with cluster centroids
            new_emb = nn.Embedding(
                k + 1, model.embedding.embedding_dim, padding_idx=0
            ).to(device)
            new_emb.weight.data[1:] = torch.tensor(
                km.cluster_centers_, dtype=torch.float32, device=device
            )
            model.embedding = new_emb
            # rebuild loaders with new mapping
            train_loader, dev_loader = mkloader(tok2cl)
            kmeans_done = True
            print(">>> clustering finished, new vocab size =", k)
    return


# ---------- run a quick hyper-param grid ----- #
for lr in [0.0001, 0.00015, 0.0002]:
    print(f"\n### Training run with lr={lr} ###")
    train(max_epochs=5, lr=lr, batch=48)

# ---------- save everything --------------- #
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
