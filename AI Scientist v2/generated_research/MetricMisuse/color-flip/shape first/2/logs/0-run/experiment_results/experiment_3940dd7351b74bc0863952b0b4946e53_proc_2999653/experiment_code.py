import os, pathlib, random, math, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------------- working directory -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- experiment container ----------------
experiment_data = {
    "contrastive_pretrain": {"loss": []},
    "supervised_finetune": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# ---------------- reproducibility -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- device ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- SPR_BENCH locating ------------------
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
        if (p / "train.csv").exists() and (p / "dev.csv").exists():
            print("Found SPR_BENCH at:", p)
            return p
    raise FileNotFoundError("SPR_BENCH with csv files not found.")


DATA_PATH = find_spr_bench_path()


# -------------- dataset utils -----------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _one(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    out = DatasetDict()
    for split in ["train", "dev", "test"]:
        out[split] = _one(f"{split}.csv")
    return out


def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def combined_complexity_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------------- vocab & labels --------------------
def build_vocab(dataset) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset) -> Dict[str, int]:
    labels = sorted({ex["label"] for ex in dataset})
    return {lab: i for i, lab in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print(f"Vocab size: {len(vocab)}, num_labels: {num_labels}")


# -------------- augmentation funcs ------------------
def augment_sequence(seq: str) -> str:
    toks = seq.split()
    # 30% token mask
    for i in range(len(toks)):
        if random.random() < 0.3:
            toks[i] = "<unk>"
    # 30% local shuffle (swap neighbour tokens)
    if len(toks) > 1 and random.random() < 0.3:
        idx = random.randint(0, len(toks) - 2)
        toks[idx], toks[idx + 1] = toks[idx + 1], toks[idx]
    return " ".join(toks)


# -------------- torch datasets ----------------------
class SPRContrastiveDataset(torch.utils.data.Dataset):
    """return two augmented views of a sequence"""

    def __init__(self, hf_ds, vocab):
        self.data = hf_ds
        self.vocab = vocab

    def encode(self, seq: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]["sequence"]
        view1 = self.encode(augment_sequence(seq))
        view2 = self.encode(augment_sequence(seq))
        return {
            "view1": torch.tensor(view1, dtype=torch.long),
            "view2": torch.tensor(view2, dtype=torch.long),
        }


class SPRSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, vocab, label2id):
        self.data = hf_ds
        self.vocab = vocab
        self.label2id = label2id

    def encode(self, seq: str):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_contrastive(batch):
    # Flatten views: [B*2, L]
    views = [b["view1"] for b in batch] + [b["view2"] for b in batch]
    max_len = max(v.size(0) for v in views)
    out = torch.full((len(views), max_len), pad_id, dtype=torch.long)
    for i, v in enumerate(views):
        out[i, : v.size(0)] = v
    return {"input_ids": out}


def collate_supervised(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    seqs = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = b["input_ids"]
        labels[i] = b["label"]
        seqs.append(b["sequence"])
    return {"input_ids": input_ids, "labels": labels, "sequences": seqs}


# ----------------- model ----------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        mean = summed / lengths
        z = self.proj(mean)
        return z


class Classifier(nn.Module):
    def __init__(self, encoder: SequenceEncoder, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.proj[-1].out_features, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# ----------------- contrastive loss -----------------
def nt_xent_loss(z, temp=0.5):
    """z: (2B, d) tensor normalized"""
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp  # (2B,2B)
    B = z.size(0) // 2
    labels = torch.arange(0, 2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    # mask self similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


# -------------- training functions ------------------
def run_contrastive_pretrain(encoder, dataset, epochs=5, batch_size=256, lr=1e-3):
    encoder.train()
    encoder.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_contrastive,
        num_workers=0,
    )
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            z = encoder(batch["input_ids"])
            loss = nt_xent_loss(z)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch["input_ids"].size(0)
        epoch_loss /= len(dataset) * 2  # two views per sample
        experiment_data["contrastive_pretrain"]["loss"].append(
            {"epoch": epoch, "loss": epoch_loss}
        )
        print(f"Contrastive Epoch {epoch}: loss={epoch_loss:.4f}")


def run_supervised_finetune(
    encoder,
    train_ds,
    dev_ds,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    patience=3,
):
    model = Classifier(encoder, num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_supervised,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_supervised,
        num_workers=0,
    )

    best_ccwa, epochs_no_imp = -1, 0
    for epoch in range(1, epochs + 1):
        # -------- train ----------
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch_t["input_ids"])
            loss = criterion(logits, batch_t["labels"])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch_t["labels"].size(0)
        tr_loss /= len(train_ds)
        experiment_data["supervised_finetune"]["losses"]["train"].append(tr_loss)

        # -------- eval -----------
        model.eval()
        dev_loss, all_pred, all_true, all_seq = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch_t = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch_t["input_ids"])
                loss = criterion(logits, batch_t["labels"])
                dev_loss += loss.item() * batch_t["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                truths = batch_t["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(truths)
                all_seq.extend(batch["sequences"])
        dev_loss /= len(dev_ds)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        ccwa = combined_complexity_weighted_accuracy(all_seq, all_true, all_pred)
        experiment_data["supervised_finetune"]["losses"]["val"].append(dev_loss)
        experiment_data["supervised_finetune"]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "swa": swa,
                "cwa": cwa,
                "ccwa": ccwa,
                "loss": dev_loss,
            }
        )
        experiment_data["supervised_finetune"]["metrics"]["train"].append(
            {"epoch": epoch, "loss": tr_loss}
        )
        print(
            f"Epoch {epoch}: val_loss={dev_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} CCWA={ccwa:.4f}"
        )

        # early stopping on CCWA
        if ccwa > best_ccwa + 1e-5:
            best_ccwa = ccwa
            best_state = model.state_dict()
            epochs_no_imp = 0
            experiment_data["supervised_finetune"]["predictions"] = all_pred
            experiment_data["supervised_finetune"]["ground_truth"] = all_true
        else:
            epochs_no_imp += 1
        if epochs_no_imp >= patience:
            print("Early stopping triggered.")
            break
    # load best
    model.load_state_dict(best_state)


# ------------------- run pipeline -------------------
# Build datasets
contrastive_ds = SPRContrastiveDataset(spr["train"], vocab)
train_ds = SPRSupervisedDataset(spr["train"], vocab, label2id)
dev_ds = SPRSupervisedDataset(spr["dev"], vocab, label2id)

# Initialise encoder
encoder = SequenceEncoder(len(vocab), emb_dim=64, hidden_dim=128, pad_idx=pad_id)

# 1. Contrastive pretrain
run_contrastive_pretrain(encoder, contrastive_ds, epochs=5)

# 2. Supervised fine-tune
run_supervised_finetune(encoder, train_ds, dev_ds, epochs=15)

# --------------- save experiment data --------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
