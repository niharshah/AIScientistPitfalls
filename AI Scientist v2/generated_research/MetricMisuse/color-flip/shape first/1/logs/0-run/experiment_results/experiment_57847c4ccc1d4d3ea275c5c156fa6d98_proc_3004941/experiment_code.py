import os, pathlib, random, gc, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset as HFDataset

# --------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- utils ---------------
def tokenize(seq: str):
    return seq.strip().split()


def count_shape_variety(seq: str):
    return len({tok[0] for tok in tokenize(seq)})


def count_color_variety(seq: str):
    return len({tok[1] for tok in tokenize(seq)})


def scwa(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) * count_color_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / sum(w) if sum(w) else 0.0


# -------- dataset loading -----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    out = DatasetDict()
    for sp in ["train", "dev", "test"]:
        out[sp] = _load(f"{sp}.csv")
    return out


data_root_candidates = [pathlib.Path(p) for p in ["SPR_BENCH", "./data/SPR_BENCH"]]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print("Loaded real SPR_BENCH from", p)
        break

# ------ synthetic fallback ----------
if spr_bench is None:
    print("SPR_BENCH not found – creating synthetic fallback.")
    shapes, colors = "ABCD", "1234"

    def synth(n):
        seqs, labels = [], []
        for i in range(n):
            L = random.randint(4, 9)
            seqs.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 600), ("dev", 150), ("test", 150)]:
        spr_bench[split] = HFDataset.from_dict(synth(n))

# -------- build vocab --------------
vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
mask_id = vocab["<MASK>"]
print("Vocab size:", vocab_size)


# ---------- dataset classes ----------
class ContrastiveNoAugDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return toks, toks  # identical views


class ClassifierDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


# ---------- collate fn --------------
def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def encode(toks_list):
        ids = [
            torch.tensor([vocab[t] for t in toks], dtype=torch.long)
            for toks in toks_list
        ]
        lens = torch.tensor([len(x) for x in ids], dtype=torch.long)
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        return ids, lens

    ids1, l1 = encode(v1)
    ids2, l2 = encode(v2)
    return {"ids1": ids1, "len1": l1, "ids2": ids2, "len2": l2}


def collate_classifier(batch):
    toks, labs, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in tk], dtype=torch.long) for tk in toks]
    lens = torch.tensor([len(x) for x in ids], dtype=torch.long)
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return {
        "ids": ids,
        "len": lens,
        "label": torch.tensor(labs, dtype=torch.long),
        "sequence": raw,
    }


# -------------- model ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, n_cls):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, n_cls)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# ------------- NT-Xent --------------
def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    sim.fill_diagonal_(-9e15)
    logits12 = sim[:N, N:]
    logits21 = sim[N:, :N]
    loss = (
        -logits12.diag()
        + torch.logsumexp(sim[:N], 1)
        - logits21.diag()
        + torch.logsumexp(sim[N:], 1)
    ).mean() * 0.5
    return loss


# -------- experiment container -------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"SWA": [], "CWA": [], "SCWA": []},
        "losses": {"pretrain": [], "train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# ----------- training settings -------
epoch_settings = [2, 4, 6, 8, 10]  # pre-training epochs grid
ft_epochs = 3  # fine-tuning epochs

# -------- iterate grid ---------------
for pte in epoch_settings:
    print(f"\n=== Pre-training epochs = {pte} ===")
    # datasets / loaders
    pretrain_loader = DataLoader(
        ContrastiveNoAugDataset(spr_bench["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_contrastive,
    )
    train_loader = DataLoader(
        ClassifierDataset(spr_bench["train"]),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_classifier,
    )
    dev_loader = DataLoader(
        ClassifierDataset(spr_bench["dev"]),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_classifier,
    )

    # models
    encoder = Encoder(vocab_size).to(device)
    clf = Classifier(encoder, len(set(spr_bench["train"]["label"]))).to(device)

    # ----- contrastive pre-training -----
    opt_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for ep in range(1, pte + 1):
        encoder.train()
        running = 0.0
        for batch in pretrain_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt_pt.zero_grad()
            z1 = encoder(batch["ids1"], batch["len1"])
            z2 = encoder(batch["ids2"], batch["len2"])
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt_pt.step()
            running += loss.item() * batch["ids1"].size(0)
        running /= len(pretrain_loader.dataset)
        experiment_data["SPR_BENCH"]["losses"]["pretrain"].append(running)
        print(f"  Pre-train epoch {ep}/{pte}: loss={running:.4f}")

    # -------- fine-tuning classifier ----------
    opt_ft = torch.optim.Adam(clf.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, ft_epochs + 1):
        # --- train ---
        clf.train()
        tr_loss = 0.0
        for b in train_loader:
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            opt_ft.zero_grad()
            logits = clf(b["ids"], b["len"])
            l = crit(logits, b["label"])
            l.backward()
            opt_ft.step()
            tr_loss += l.item() * b["ids"].size(0)
        tr_loss /= len(train_loader.dataset)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # --- validation ---
        clf.eval()
        val_loss = 0.0
        preds, gts, seqs = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                b = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in b.items()
                }
                logits = clf(b["ids"], b["len"])
                l = crit(logits, b["label"])
                val_loss += l.item() * b["ids"].size(0)
                preds += logits.argmax(1).cpu().tolist()
                gts += b["label"].cpu().tolist()
                seqs += b["sequence"]
        val_loss /= len(dev_loader.dataset)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)

        swa = shape_weighted_accuracy(seqs, gts, preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        scw = scwa(seqs, gts, preds)
        experiment_data["SPR_BENCH"]["metrics"]["SWA"].append(swa)
        experiment_data["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
        experiment_data["SPR_BENCH"]["metrics"]["SCWA"].append(scw)
        experiment_data["SPR_BENCH"]["predictions"].append(preds)
        experiment_data["SPR_BENCH"]["ground_truth"].append(gts)
        experiment_data["SPR_BENCH"]["epochs"].append({"pretrain": pte, "ft": ep})

        print(
            f"  FT epoch {ep}/{ft_epochs}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} SCWA={scw:.3f}"
        )

    # ---- plotting loss curves ----
    plt.figure()
    plt.plot(
        experiment_data["SPR_BENCH"]["losses"]["train"][-ft_epochs:], label="train"
    )
    plt.plot(experiment_data["SPR_BENCH"]["losses"]["val"][-ft_epochs:], label="val")
    plt.title(f"Fine-tune loss (pretrain={pte})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_noaug_{pte}.png"))
    plt.close()

    # cleanup gpu
    del encoder, clf, opt_pt, opt_ft
    torch.cuda.empty_cache()
    gc.collect()

# -------- save experiment -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. Data and plots saved in ./working/")
