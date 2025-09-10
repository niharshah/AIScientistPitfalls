import os, pathlib, random, math, time, json, gc
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict

# ---------------------------- misc & dirs ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------- data utils -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(1, sum(w))


# ---------------------------- dataset fetch / fallback --------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print("Loaded real SPR_BENCH from", p)
        break
if spr_bench is None:
    print("SPR_BENCH not found, generating synthetic toy dataset.")

    def synth(n):
        shapes, colors = "ABCD", "1234"
        seqs, labels = [], []
        for _ in range(n):
            L = random.randint(4, 9)
            seqs.append(
                " ".join(
                    random.choice(shapes) + random.choice(colors) for _ in range(L)
                )
            )
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# ---------------------------- vocab ---------------------------------
def tokenize(s):
    return s.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size = len(vocab)
mask_id = vocab["<MASK>"]
print("Vocab size =", vocab_size)


# ---------------------------- torch datasets ------------------------
class SPRContrastiveDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs = hf_ds["sequence"]

    def __len__(self):
        return len(self.seqs)

    def _augment(self, toks, p=0.15):
        out = [t for t in toks if random.random() >= p]
        if not out:
            out.append(random.choice(toks))
        return out

    def __getitem__(self, idx):
        toks = tokenize(self.seqs[idx])
        return self._augment(toks), self._augment(toks)


class SPRClassifierDataset(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return tokenize(self.seqs[idx]), self.labels[idx], self.seqs[idx]


def collate_contrastive(batch):
    v1, v2 = zip(*batch)

    def encode(tok_lists):
        ids = [torch.tensor([vocab[t] for t in toks]) for toks in tok_lists]
        lens = [len(x) for x in ids]
        return pad_sequence(ids, batch_first=True), torch.tensor(lens)

    ids1, len1 = encode(v1)
    ids2, len2 = encode(v2)
    return {
        "ids1": ids1.to(device),
        "len1": len1.to(device),
        "ids2": ids2.to(device),
        "len2": len2.to(device),
    }


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in t]) for t in toks]
    lens = [len(i) for i in ids]
    return {
        "ids": pad_sequence(ids, batch_first=True).to(device),
        "len": torch.tensor(lens).to(device),
        "label": torch.tensor(labels).to(device),
        "sequence": raw,
    }


pretrain_loader = DataLoader(
    SPRContrastiveDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_contrastive,
)
train_loader = DataLoader(
    SPRClassifierDataset(spr_bench["train"]),
    batch_size=128,
    shuffle=True,
    collate_fn=collate_classifier,
)
dev_loader = DataLoader(
    SPRClassifierDataset(spr_bench["dev"]),
    batch_size=256,
    shuffle=False,
    collate_fn=collate_classifier,
)


# ---------------------------- model defs ----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
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
    def __init__(self, enc, num_classes):
        super().__init__()
        self.encoder = enc
        self.head = nn.Linear(enc.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


def nt_xent_loss(z1, z2, temp=0.5):
    z1 = nn.functional.normalize(z1, 1)
    z2 = nn.functional.normalize(z2, 1)
    N = z1.size(0)
    z = torch.cat([z1, z2])
    sim = (z @ z.T) / temp
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


# ---------------------------- experiment dict -----------------------
experiment_data = {"learning_rate": {}}

# ---------------------------- training loop ------------------------
lr_grid = [3e-4, 1e-3, 3e-3]
pretrain_epochs, ft_epochs = 2, 3

for lr in lr_grid:
    key = str(lr)
    experiment_data["learning_rate"][key] = {
        "SPR_BENCH": {
            "losses": {"pretrain": [], "train": [], "val": []},
            "metrics": {"SWA": [], "CWA": [], "SCHM": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    # fresh models
    encoder = Encoder(vocab).to(device)
    clf = Classifier(encoder, num_classes=len(set(spr_bench["train"]["label"]))).to(
        device
    )
    # ---------- pretrain ----------
    opt_pt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for ep in range(1, pretrain_epochs + 1):
        encoder.train()
        running = 0.0
        for batch in pretrain_loader:
            opt_pt.zero_grad()
            z1 = encoder(batch["ids1"], batch["len1"])
            z2 = encoder(batch["ids2"], batch["len2"])
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            opt_pt.step()
            running += loss.item() * batch["ids1"].size(0)
        epoch_loss = running / len(pretrain_loader.dataset)
        experiment_data["learning_rate"][key]["SPR_BENCH"]["losses"]["pretrain"].append(
            epoch_loss
        )
        print(f"[lr={lr}] Pretrain {ep}/{pretrain_epochs}: loss={epoch_loss:.4f}")

    # ---------- fine-tune ----------
    opt_ft = torch.optim.Adam(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, ft_epochs + 1):
        clf.train()
        run_loss = 0.0
        for batch in train_loader:
            opt_ft.zero_grad()
            logits = clf(batch["ids"], batch["len"])
            loss = ce(logits, batch["label"])
            loss.backward()
            opt_ft.step()
            run_loss += loss.item() * batch["ids"].size(0)
        train_loss = run_loss / len(train_loader.dataset)
        # eval
        clf.eval()
        val_loss = 0.0
        preds = true = seqs = []
        all_pred, all_true, all_seq = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                logits = clf(batch["ids"], batch["len"])
                loss = ce(logits, batch["label"])
                val_loss += loss.item() * batch["ids"].size(0)
                p = logits.argmax(1).cpu().tolist()
                all_pred += p
                all_true += batch["label"].cpu().tolist()
                all_seq += batch["sequence"]
        val_loss /= len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) > 0 else 0.0
        exp = experiment_data["learning_rate"][key]["SPR_BENCH"]
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["SWA"].append(swa)
        exp["metrics"]["CWA"].append(cwa)
        exp["metrics"]["SCHM"].append(schm)
        if ep == ft_epochs:
            exp["predictions"] = all_pred
            exp["ground_truth"] = all_true
        print(
            f"[lr={lr}] FT {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
        )
    # plot per lr
    plt.figure()
    plt.plot(exp["losses"]["train"], label="train")
    plt.plot(exp["losses"]["val"], label="val")
    plt.title(f"Fine-tune loss (lr={lr})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_lr_{key.replace('.','p')}.png"))
    plt.close()
    # cleanup
    del encoder, clf, opt_pt, opt_ft
    torch.cuda.empty_cache()
    gc.collect()

# ------------------------ save experiment data ----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy and loss curves in ./working/")
