import os, pathlib, random, time, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ------- working dir & container -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ---------------- GPU / CPU -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- reproducibility -------------------------------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------- dataset utilities ----------------------------------------------
def find_spr_bench_path() -> pathlib.Path:
    cands = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cands:
        if c and (pathlib.Path(c) / "train.csv").exists():
            return pathlib.Path(c).resolve()
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench_path()


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv",
            data_files=str(root / csv),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


def count_shape_variety(seq: str) -> int:
    return len(set(t[0] for t in seq.split() if t))


def count_color_variety(seq: str) -> int:
    return len(set(t[1] for t in seq.split() if len(t) > 1))


def complexity_weight(seq: str) -> int:
    return count_shape_variety(seq) + count_color_variety(seq)


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if w else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if w else 0.0


def ccwa(seqs, y_t, y_p):
    w = [complexity_weight(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if w else 0.0


# ---------------- vocab & label maps ---------------------------------------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labs = sorted({ex["label"] for ex in dataset})
    return {l: i for i, l in enumerate(labs)}


vocab = build_vocab(spr["train"])
pad_id = vocab["<pad>"]
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
print(f"Vocab {len(vocab)} , Labels {len(label2id)}")


# ---------------- torch Dataset ---------------------------------------------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hfset, vocab, label2id):
        self.data = hfset
        self.vocab = vocab
        self.label2id = label2id

    def encode(self, seq):
        return [self.vocab.get(t, 1) for t in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    seqs = [b["sequence"] for b in batch]
    for i, b in enumerate(batch):
        input_ids[i, : len(b["input_ids"])] = b["input_ids"]
    return {"input_ids": input_ids, "labels": labels, "sequences": seqs}


train_ds = SPRTorchDataset(spr["train"], vocab, label2id)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id)


# ---------------- model -----------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=256, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid, bidirectional=True, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        return summed / lengths  # mean-pooled representation


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.lstm.hidden_size * 2, num_labels)

    def forward(self, x):
        return self.fc(self.encoder(x))


# ---------------- simple token augmentation --------------------------------------
def augment_tokens(tokens):
    # random deletion
    new = [t for t in tokens if random.random() > 0.2]
    if not new:
        new = [random.choice(tokens)]
    # small shuffle
    if len(new) > 2 and random.random() > 0.5:
        i = random.randrange(len(new) - 1)
        new[i], new[i + 1] = new[i + 1], new[i]
    return new


def augment_encode(seq, vocab):
    tokens = seq.split()
    aug = augment_tokens(tokens)
    return [vocab.get(t, 1) for t in aug]


# ---------------- contrastive pre-training ---------------------------------------
def contrastive_pretrain(encoder, epochs=3, batch_size=256, tau=0.1):
    enc = encoder.to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=0,
    )
    for ep in range(1, epochs + 1):
        enc.train()
        total_loss = 0
        n = 0
        for batch in loader:
            # build two augmented views
            view1 = [augment_encode(b["sequence"], vocab) for b in batch]
            view2 = [augment_encode(b["sequence"], vocab) for b in batch]

            # pad
            def pad_to_mat(lst):
                ml = max(len(t) for t in lst)
                mat = torch.full((len(lst), ml), pad_id, dtype=torch.long)
                for i, tok in enumerate(lst):
                    mat[i, : len(tok)] = torch.tensor(tok)
                return mat.to(device)

            x1, x2 = pad_to_mat(view1), pad_to_mat(view2)
            z1 = F.normalize(enc(x1), dim=1)
            z2 = F.normalize(enc(x2), dim=1)
            z = torch.cat([z1, z2], 0)  # 2N,d
            sim = z @ z.T  # cosine sim (since normalized)
            N = z1.size(0)
            mask = torch.eye(2 * N, dtype=torch.bool, device=device)
            sim.masked_fill_(mask, -9e15)  # exclude self-similarities
            sim /= tau
            # labels: positives are diagonal offset by N
            positives = torch.arange(N, 2 * N, device=device)
            logits = torch.cat([sim[:N], sim[N:]], 0)
            labels = torch.cat([positives, torch.arange(0, N, device=device)], 0)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * (2 * N)
            n += 2 * N
        print(f"Contrastive epoch {ep}: loss={total_loss/n:.4f}")


# ---------------- fine-tune classification ---------------------------------------
def train_classifier(max_epochs=15, patience=3):
    encoder = Encoder(len(vocab), 128, 256, pad_idx=pad_id)
    contrastive_pretrain(encoder, epochs=3)
    model = Classifier(encoder, len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    best_ccwa = -1
    no_imp = 0
    best_state = None
    for ep in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        tot = 0
        tloss = 0.0
        for batch in tr_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            opt.step()
            tloss += loss.item() * batch["labels"].size(0)
            tot += batch["labels"].size(0)
        train_loss = tloss / tot
        # ---- val ----
        model.eval()
        vloss = 0.0
        tot = 0
        all_pred = []
        all_true = []
        all_seq = []
        with torch.no_grad():
            for batch in val_loader:
                tbatch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(tbatch["input_ids"])
                loss = crit(logits, tbatch["labels"])
                vloss += loss.item() * tbatch["labels"].size(0)
                tot += tbatch["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                trues = tbatch["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(trues)
                all_seq.extend(batch["sequences"])
        val_loss = vloss / tot
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        cc = ccwa(all_seq, all_true, all_pred)
        hwa = 2 * swa * cwa / (swa + cwa) if swa + cwa > 0 else 0.0
        print(f"Epoch {ep}: validation_loss = {val_loss:.4f} CCWA={cc:.4f}")
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            {"epoch": ep, "train_loss": train_loss}
        )
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {
                "epoch": ep,
                "swa": swa,
                "cwa": cwa,
                "ccwa": cc,
                "hwa": hwa,
                "val_loss": val_loss,
            }
        )
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        if cc > best_ccwa + 1e-5:
            best_ccwa = cc
            best_state = model.state_dict()
            no_imp = 0
            experiment_data["SPR_BENCH"]["predictions"] = all_pred
            experiment_data["SPR_BENCH"]["ground_truth"] = all_true
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state:
        model.load_state_dict(best_state)


train_classifier()

# ---------------- save experiment data -------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("saved experiment_data.npy")
