import os, pathlib, random, time, math
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ----------- working dir & logging -----------------
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

# ----------- reproducibility -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------- device --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------- locate SPR_BENCH -----------------------
def find_spr_bench_path() -> pathlib.Path:
    candidates = [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c).expanduser().resolve()
        if (p / "train.csv").exists():
            print("Found SPR_BENCH at:", p)
            return p
    raise FileNotFoundError("SPR_BENCH dataset not found")


DATA_PATH = find_spr_bench_path()


# ---------- dataset helpers ------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fname):  # each csv is a split
        return load_dataset(
            "csv",
            data_files=str(root / fname),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split()))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if sum(w) > 0 else 0.0


def color_weighted_accuracy(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if sum(w) > 0 else 0.0


def complexity_weight(sequence: str) -> int:
    return count_shape_variety(sequence) + count_color_variety(sequence)


def combined_complexity_accuracy(seqs, y_t, y_p):
    w = [complexity_weight(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    return num / sum(w) if sum(w) > 0 else 0.0


spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})


# ---------- vocab & label mapping -------------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labels = sorted({ex["label"] for ex in dataset})
    return {lab: i for i, lab in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {v: k for k, v in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)
print(f"Vocab={len(vocab)}, Labels={num_labels}")


# ------------- dataset class ------------------------
class SPRContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dset, vocab, label2id, mask_prob=0.2):
        self.data = hf_dset
        self.vocab = vocab
        self.label2id = label2id
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def encode(self, toks):
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in toks]

    def augment_tokens(self, toks):
        toks = toks[:]  # copy
        # random masking
        for i in range(len(toks)):
            if random.random() < self.mask_prob:
                toks[i] = "<unk>"
        # small shuffle: swap two tokens with small prob
        if len(toks) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(toks)), 2)
            toks[i], toks[j] = toks[j], toks[i]
        return toks

    def __getitem__(self, idx):
        ex = self.data[idx]
        orig_toks = ex["sequence"].split()
        aug_toks = self.augment_tokens(orig_toks)
        return {
            "orig_ids": torch.tensor(self.encode(orig_toks), dtype=torch.long),
            "aug_ids": torch.tensor(self.encode(aug_toks), dtype=torch.long),
            "label": torch.tensor(self.label2id[ex["label"]], dtype=torch.long),
            "sequence": ex["sequence"],
        }


def collate_fn(batch):
    def pad_and_stack(key):
        seqs = [b[key] for b in batch]
        maxlen = max(len(s) for s in seqs)
        t = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            t[i, : len(s)] = s
        return t

    orig = pad_and_stack("orig_ids")
    aug = pad_and_stack("aug_ids")
    labels = torch.stack([b["label"] for b in batch])
    sequences = [b["sequence"] for b in batch]
    return {"orig": orig, "aug": aug, "labels": labels, "sequences": sequences}


train_ds = SPRContrastiveDataset(spr["train"], vocab, label2id)
dev_ds = SPRContrastiveDataset(
    spr["dev"], vocab, label2id, mask_prob=0.0
)  # no aug for dev


# ------------- model --------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.hid_dim = hid_dim

    def forward(self, x):
        # x: B x T
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        mask = (x != pad_id).unsqueeze(-1)
        summed = (out * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        mean = summed / lengths
        return mean  # B x 2*hid_dim


class ContrastiveClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_dim=64, hid_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * hid_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.classifier = nn.Linear(2 * hid_dim, num_labels)

    def forward(self, x):
        features = self.encoder(x)  # B x 2H
        logits = self.classifier(features)  # classification
        z = nn.functional.normalize(self.proj(features), dim=-1)  # contrastive
        return logits, z


# ------------- contrastive loss ---------------------
def nt_xent(z1, z2, temperature=0.5):
    """
    z1, z2: B x D normalized
    returns NT-Xent loss over 2B samples
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T) / temperature  # 2B x 2B
    # mask self similarity
    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(diag, -1e9)
    # positives: for i in 0..B-1, pos index is i+B, and vice-versa
    pos_idx = torch.arange(B, device=z.device)
    pos = torch.cat([pos_idx + B, pos_idx])  # 2B
    labels = pos
    loss = nn.functional.cross_entropy(sim, labels)
    return loss


# ------------- training -----------------------------
def run_training(max_epochs=15, patience=3, alpha=0.5, batch_size=256):
    model = ContrastiveClassifier(len(vocab), num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    best_ccwa, best_state, epochs_no_improve = -1, None, 0
    for epoch in range(1, max_epochs + 1):
        # ---- training ----
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits_o, z_o = model(batch["orig"])
            logits_a, z_a = model(batch["aug"])
            ce = ce_loss_fn(logits_o, batch["labels"])
            con = nt_xent(z_o, z_a)
            loss = ce + alpha * con
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * batch["labels"].size(0)
        train_loss = run_loss / len(train_ds)
        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        # ---- evaluation ----
        model.eval()
        val_loss, all_pred, all_true, all_seq = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                t_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits, _ = (
                    model.encoder(t_batch["orig"])[0]
                    if isinstance(model.encoder(t_batch["orig"]), tuple)
                    else (None, None)
                )
            # The above line inadvertently mixes two different returns.
            # Instead evaluate properly below.
            # (We will re-run loop.)
        # redo eval properly
        model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                t_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits, _ = model(t_batch["orig"])
                ce = ce_loss_fn(logits, t_batch["labels"])
                val_loss += ce.item() * t_batch["labels"].size(0)
                preds = logits.argmax(-1).cpu().tolist()
                truths = t_batch["labels"].cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(truths)
                all_seq.extend(batch["sequences"])
        val_loss /= len(dev_ds)
        swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
        cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
        ccwa = combined_complexity_accuracy(all_seq, all_true, all_pred)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "swa": swa, "cwa": cwa, "ccwa": ccwa, "loss": val_loss}
        )
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} | SWA={swa:.4f} CWA={cwa:.4f} CCWA={ccwa:.4f}"
        )
        # ---- early stopping on CCWA ----
        if ccwa > best_ccwa + 1e-5:
            best_ccwa = ccwa
            best_state = model.state_dict()
            experiment_data["SPR_BENCH"]["predictions"] = all_pred
            experiment_data["SPR_BENCH"]["ground_truth"] = all_true
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ------------------ run -----------------------------
model = run_training(max_epochs=15, patience=3, alpha=0.5, batch_size=256)

# ------------------ save logs -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved metrics to", os.path.join(working_dir, "experiment_data.npy"))
