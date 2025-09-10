import os, pathlib, random, math, json, time
import numpy as np
import torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datasets import load_dataset, DatasetDict

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------------------------
# -------- SPR helpers ----------------------------
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


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


# -------------------------------------------------
# --------- Load / synthesize data ----------------
data_root_candidates = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("./data/SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
spr_bench = None
for p in data_root_candidates:
    if (p / "train.csv").exists():
        spr_bench = load_spr_bench(p)
        print("Loaded dataset from", p)
        break
if spr_bench is None:
    print("SPR_BENCH not found, generating tiny synthetic data.")

    def synth(n):
        seqs, labels = [], []
        shapes, colors = "ABCD", "1234"
        for _ in range(n):
            L = random.randint(4, 9)
            seq = " ".join(
                random.choice(shapes) + random.choice(colors) for _ in range(L)
            )
            seqs.append(seq)
            labels.append(random.randint(0, 1))
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr_bench = DatasetDict()
    for split, n in [("train", 500), ("dev", 100), ("test", 100)]:
        spr_bench[split] = load_dataset(
            "json", data_files={"train": None}, split="train", data=synth(n)
        )


# -------------------------------------------------
# --------- Vocabulary ----------------------------
def tokenize(seq):
    return seq.strip().split()


vocab = {"<PAD>": 0, "<MASK>": 1}
for split in spr_bench.values():
    for seq in split["sequence"]:
        for tok in tokenize(seq):
            if tok not in vocab:
                vocab[tok] = len(vocab)
vocab_size, mask_id = len(vocab), vocab["<MASK>"]
print("Vocab size =", vocab_size)


# -------------------------------------------------
# --------- Datasets / loaders --------------------
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

    def encode(list_tokens):
        ids = [torch.tensor([vocab[t] for t in toks]) for toks in list_tokens]
        lens = torch.tensor([len(i) for i in ids])
        ids = pad_sequence(ids, batch_first=True, padding_value=0)
        return ids.to(device), lens.to(device)

    ids1, len1 = encode(v1)
    ids2, len2 = encode(v2)
    return {"ids1": ids1, "len1": len1, "ids2": ids2, "len2": len2}


def collate_classifier(batch):
    toks, labels, raw = zip(*batch)
    ids = [torch.tensor([vocab[t] for t in tk]) for tk in toks]
    lens = torch.tensor([len(i) for i in ids])
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return {
        "ids": ids.to(device),
        "len": lens.to(device),
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


# -------------------------------------------------
# --------- Models --------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, ids, lens):
        x = self.emb(ids)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.proj(h)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_classes)

    def forward(self, ids, lens):
        return self.head(self.encoder(ids, lens))


# -------------------------------------------------
# --------- Loss utilities ------------------------
def nt_xent_loss(z1, z2, temp=0.5):
    z1, z2 = nn.functional.normalize(z1, 1), nn.functional.normalize(z2, 1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = torch.matmul(z, z.T) / temp
    sim.fill_diagonal_(-9e15)
    pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(device)
    loss = nn.functional.cross_entropy(sim, pos)
    return loss


# -------------------------------------------------
# --------- Hyper-parameter sweep -----------------
finetune_lrs = [1e-3, 5e-4, 3e-4, 1e-4]  # values to try
experiment_data = {"learning_rate": {"SPR_BENCH": {"runs": []}}}

for lr_ft in finetune_lrs:
    print(f"\n============= Fine-tune LR = {lr_ft:.1e} =============")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    encoder = Encoder(vocab_size).to(device)
    clf_model = Classifier(
        encoder, num_classes=len(set(spr_bench["train"]["label"]))
    ).to(device)

    record = {
        "lr_ft": lr_ft,
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "SCHM": []},
        "predictions": [],
        "ground_truth": [],
    }

    # ---- pre-training ----
    optimizer_pt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for ep in range(1, 3):
        encoder.train()
        ep_loss = 0.0
        for b in pretrain_loader:
            optimizer_pt.zero_grad()
            z1, z2 = encoder(b["ids1"], b["len1"]), encoder(b["ids2"], b["len2"])
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer_pt.step()
            ep_loss += loss.item() * b["ids1"].size(0)
        ep_loss /= len(pretrain_loader.dataset)
        record["losses"]["pretrain"].append(ep_loss)
        print(f"  Pretrain  ep{ep}: loss={ep_loss:.4f}")

    # ---- fine-tuning ----
    optimizer_ft = torch.optim.Adam(clf_model.parameters(), lr=lr_ft)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, 4):
        # train
        clf_model.train()
        run_loss = 0.0
        for b in train_loader:
            optimizer_ft.zero_grad()
            logits = clf_model(b["ids"], b["len"])
            loss = ce(logits, b["label"])
            loss.backward()
            optimizer_ft.step()
            run_loss += loss.item() * b["ids"].size(0)
        tr_loss = run_loss / len(train_loader.dataset)
        record["losses"]["train"].append(tr_loss)

        # val
        clf_model.eval()
        v_loss = 0.0
        all_p, all_t, all_s = [], [], []
        with torch.no_grad():
            for b in dev_loader:
                logits = clf_model(b["ids"], b["len"])
                loss = ce(logits, b["label"])
                v_loss += loss.item() * b["ids"].size(0)
                preds = logits.argmax(1).cpu().tolist()
                all_p += preds
                all_t += b["label"].cpu().tolist()
                all_s += b["sequence"]
        v_loss /= len(dev_loader.dataset)
        record["losses"]["val"].append(v_loss)

        swa = shape_weighted_accuracy(all_s, all_t, all_p)
        cwa = color_weighted_accuracy(all_s, all_t, all_p)
        schm = 2 * swa * cwa / (swa + cwa) if (swa + cwa) else 0.0
        record["metrics"]["SWA"].append(swa)
        record["metrics"]["CWA"].append(cwa)
        record["metrics"]["SCHM"].append(schm)
        if ep == 3:  # store predictions only once
            record["predictions"] = all_p
            record["ground_truth"] = all_t
        print(
            f"  Finetune ep{ep}: train={tr_loss:.4f} val={v_loss:.4f} | SWA={swa:.3f} CWA={cwa:.3f} SCHM={schm:.3f}"
        )

    # ---- plot ----
    plt.figure()
    plt.plot(record["losses"]["train"], label="train")
    plt.plot(record["losses"]["val"], label="val")
    plt.title(f"Loss (ft-lr={lr_ft:.1e})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_lr_{lr_ft:.1e}.png"))
    plt.close()

    experiment_data["learning_rate"]["SPR_BENCH"]["runs"].append(record)

# -------------------------------------------------
# --------- save all experiment data --------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved logs & plots to", working_dir)
