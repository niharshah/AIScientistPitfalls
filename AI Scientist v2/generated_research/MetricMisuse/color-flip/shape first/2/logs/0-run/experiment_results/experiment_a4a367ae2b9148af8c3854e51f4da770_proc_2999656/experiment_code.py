import os, pathlib, random, math, time
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- experiment container ----------
experiment_data = {
    "pretrain": {"loss": []},
    "finetune": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# -------- reproducibility ----------
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    cand = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in cand:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists():
            print("Found SPR_BENCH at:", p)
            return p
    raise FileNotFoundError("SPR_BENCH dataset not found.")


DATA_PATH = find_spr_bench_path()


# ------------ dataset utils -----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _one(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict(
        {split: _one(f"{split}.csv") for split in ["train", "dev", "test"]}
    )


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def CCWA(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def SWA(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def CWA(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ----------- vocab & labels ------------
SPECIAL = ["<pad>", "<unk>", "[CLS]", "[MASK]"]
vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL)}


def add_token(tok: str):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for ex in spr["train"]:
    for t in ex["sequence"].split():
        add_token(t)

pad_id, unk_id, cls_id, mask_id = (vocab[t] for t in SPECIAL)
label2id = {
    lab: i for i, lab in enumerate(sorted({ex["label"] for ex in spr["train"]}))
}
id2label = {i: l for l, i in label2id.items()}
print(f"Vocab size={len(vocab)}, num_labels={len(label2id)}")


# --------- augmentation helpers ----------
def shuffle_pair(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens
    idx = random.randint(0, len(tokens) - 2)
    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
    return tokens


def make_view(tokens: List[str]) -> List[str]:
    out = tokens.copy()
    # token masking 30%
    for i in range(len(out)):
        if random.random() < 0.3:
            out[i] = "<unk>"
    # neighbour shuffle 30%
    if random.random() < 0.3:
        out = shuffle_pair(out)
    return out


def mask_for_mlm(tokens: List[int]) -> Tuple[List[int], List[int]]:
    input_ids = tokens.copy()
    labels = [-100] * len(tokens)
    for i in range(1, len(tokens)):  # skip CLS
        if random.random() < 0.15:
            labels[i] = tokens[i]
            r = random.random()
            if r < 0.8:
                input_ids[i] = mask_id
            elif r < 0.9:
                input_ids[i] = random.randint(4, len(vocab) - 1)
            # else keep original
    return input_ids, labels


# ----------- torch datasets -------------
class SPRPretrainDS(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.data = hf_ds

    def encode(self, seq: str) -> List[int]:
        return [cls_id] + [vocab.get(t, unk_id) for t in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]["sequence"]
        tokens = seq.split()
        v1 = self.encode(" ".join(make_view(tokens)))
        v2 = self.encode(" ".join(make_view(tokens)))
        mlm_toks = self.encode(seq)
        mlm_inp, mlm_lbl = mask_for_mlm(mlm_toks)
        return {
            "v1": torch.tensor(v1),
            "v2": torch.tensor(v2),
            "mlm_in": torch.tensor(mlm_inp),
            "mlm_lbl": torch.tensor(mlm_lbl),
        }


class SPRSupervisedDS(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.data = hf_ds

    def encode(self, seq):
        return [cls_id] + [vocab.get(t, unk_id) for t in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        ids = torch.tensor(self.encode(ex["sequence"]))
        lab = torch.tensor(label2id[ex["label"]])
        return {"input_ids": ids, "label": lab, "sequence": ex["sequence"]}


def pad_batch(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    L = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), L), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_pretrain(batch):
    ids1 = pad_batch([b["v1"] for b in batch], pad_id)
    ids2 = pad_batch([b["v2"] for b in batch], pad_id)
    mlm_in = pad_batch([b["mlm_in"] for b in batch], pad_id)
    mlm_lbl = pad_batch([b["mlm_lbl"] for b in batch], -100)
    return {"v1": ids1, "v2": ids2, "mlm_in": mlm_in, "mlm_lbl": mlm_lbl}


def collate_supervised(batch):
    ids = pad_batch([b["input_ids"] for b in batch], pad_id)
    labs = torch.stack([b["label"] for b in batch])
    seqs = [b["sequence"] for b in batch]
    return {"input_ids": ids, "labels": labs, "sequences": seqs}


# ------------- model -------------------
class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, 512, emb_dim))  # assume max len 512
        encoder_layer = nn.TransformerEncoderLayer(
            emb_dim, nhead, dim_ff, dropout, batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)  # for contrastive

    def forward(self, x, return_hidden=False):
        B, L = x.size()
        emb = self.emb(x) + self.pos[:, :L, :]
        h = self.tr(emb)
        h = self.ln(h)
        cls = h[:, 0, :]
        z = self.proj(cls)
        if return_hidden:
            return z, h
        return z


class SPRClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, num_labels: int):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(encoder.proj.out_features, num_labels)

    def forward(self, x):
        z = self.enc(x)
        return self.head(z)


# ----------- losses --------------------
def nt_xent(z, temp=0.5):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    sim.fill_diagonal_(-9e15)
    return nn.functional.cross_entropy(sim, labels)


def mlm_loss(logits, labels):
    return nn.functional.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100
    )


# ----------- training loops ------------
def pretrain(encoder, ds, epochs=3, bs=64, lr=3e-4, beta=1.0):
    enc = encoder.to(device).train()
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    dl = DataLoader(
        ds, batch_size=bs, shuffle=True, collate_fn=collate_pretrain, num_workers=0
    )
    dec = nn.Linear(enc.emb.embedding_dim, len(vocab), bias=False).to(device)
    dec.weight = enc.emb.weight  # weight tying
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            z1, _ = enc(batch["v1"], return_hidden=True)
            z2, _ = enc(batch["v2"], return_hidden=True)
            z = torch.cat([z1, z2], 0)
            loss_c = nt_xent(z)
            _, hid = enc(batch["mlm_in"], return_hidden=True)
            mlm_logits = dec(hid)
            loss_m = mlm_loss(mlm_logits, batch["mlm_lbl"])
            loss = loss_c + beta * loss_m
            loss.backward()
            opt.step()
            tot_loss += loss.item() * batch["v1"].size(0)
        tot_loss /= len(ds)
        experiment_data["pretrain"]["loss"].append({"epoch": ep, "loss": tot_loss})
        print(f"Pretrain epoch {ep}: loss={tot_loss:.4f}")


def finetune(encoder, train_ds, dev_ds, epochs=10, bs=128, lr=1e-3, patience=3):
    model = SPRClassifier(encoder, len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    dl_train = DataLoader(
        train_ds, bs, True, collate_fn=collate_supervised, num_workers=0
    )
    dl_dev = DataLoader(
        dev_ds, 256, False, collate_fn=collate_supervised, num_workers=0
    )
    best_ccwa, no_imp, best_state = -1, 0, None
    for ep in range(1, epochs + 1):
        # --- train ---
        model.train()
        tr_loss = 0.0
        for batch in dl_train:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch["labels"].size(0)
        tr_loss /= len(train_ds)
        experiment_data["finetune"]["losses"]["train"].append(tr_loss)
        # --- eval ---
        model.eval()
        dev_loss, preds, truths, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dl_dev:
                bt = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(bt["input_ids"])
                loss = crit(logits, bt["labels"])
                dev_loss += loss.item() * bt["labels"].size(0)
                preds.extend(logits.argmax(-1).cpu().tolist())
                truths.extend(bt["labels"].cpu().tolist())
                seqs.extend(batch["sequences"])
        dev_loss /= len(dev_ds)
        swa, cwa, ccwa = (
            SWA(seqs, truths, preds),
            CWA(seqs, truths, preds),
            CCWA(seqs, truths, preds),
        )
        experiment_data["finetune"]["losses"]["val"].append(dev_loss)
        experiment_data["finetune"]["metrics"]["val"].append(
            {"epoch": ep, "swa": swa, "cwa": cwa, "ccwa": ccwa, "loss": dev_loss}
        )
        experiment_data["finetune"]["metrics"]["train"].append(
            {"epoch": ep, "loss": tr_loss}
        )
        print(
            f"Epoch {ep}: validation_loss = {dev_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} CCWA={ccwa:.4f}"
        )
        # early stop
        if ccwa > best_ccwa + 1e-5:
            best_ccwa = ccwa
            best_state = model.state_dict()
            no_imp = 0
            experiment_data["finetune"]["predictions"] = preds
            experiment_data["finetune"]["ground_truth"] = truths
        else:
            no_imp += 1
        if no_imp >= patience:
            print("Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)


# -------------- run pipeline -----------------
pre_ds = SPRPretrainDS(spr["train"])
train_ds = SPRSupervisedDS(spr["train"])
dev_ds = SPRSupervisedDS(spr["dev"])

encoder = TransformerEncoder(len(vocab))
pretrain(encoder, pre_ds, epochs=3)
finetune(encoder, train_ds, dev_ds, epochs=10)

# ------------- save data ---------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
