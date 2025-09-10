import os, pathlib, random, time, math, json
import numpy as np, torch
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import DataLoader

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- working dir --------------
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

# ---------- device -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- locate SPR_BENCH ----------
def find_spr_bench_path() -> pathlib.Path:
    for cand in [
        os.environ.get("SPR_BENCH_PATH", ""),
        "./SPR_BENCH",
        "../SPR_BENCH",
        "/home/zxl240011/AI-Scientist-v2/SPR_BENCH",
    ]:
        if cand:
            p = pathlib.Path(cand).expanduser().resolve()
            if (p / "train.csv").exists() and (p / "dev.csv").exists():
                return p
    raise FileNotFoundError("SPR_BENCH not found.")


DATA_PATH = find_spr_bench_path()


# ---------- dataset utils -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=str(pathlib.Path(working_dir) / ".cache_dsets"),
        )

    return DatasetDict({sp: _load(f"{sp}.csv") for sp in ["train", "dev", "test"]})


spr = load_spr_bench(DATA_PATH)


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split() if tok))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def SWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def CWA(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


def CCWA(seqs, y_t, y_p):
    w = [count_shape_variety(s) + count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_t, y_p) if t == p) / max(1, sum(w))


# ---------- vocab & labels ------------
def build_vocab(dataset):
    vocab = {"<pad>": 0, "<unk>": 1}
    for ex in dataset:
        for tok in ex["sequence"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_label_map(dataset):
    labels = sorted({ex["label"] for ex in dataset})
    return {l: i for i, l in enumerate(labels)}


vocab = build_vocab(spr["train"])
label2id = build_label_map(spr["train"])
id2label = {i: l for l, i in label2id.items()}
pad_id = vocab["<pad>"]
num_labels = len(label2id)


# ---------- torch datasets ------------
class SPRTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_set, vocab, label2id, with_label=True):
        self.data = hf_set
        self.vocab = vocab
        self.label2id = label2id
        self.with_label = with_label

    def __len__(self):
        return len(self.data)

    def encode(self, seq):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in seq.split()]

    def __getitem__(self, idx):
        ex = self.data[idx]
        item = {
            "input_ids": torch.tensor(self.encode(ex["sequence"]), dtype=torch.long),
            "sequence": ex["sequence"],
        }
        if self.with_label:
            item["label"] = torch.tensor(self.label2id[ex["label"]], dtype=torch.long)
        return item


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    seqs = []
    labels = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        ids[i, :l] = b["input_ids"]
        seqs.append(b["sequence"])
        if "label" in b:
            labels.append(b["label"])
    res = {"input_ids": ids, "sequences": seqs}
    if labels:
        res["labels"] = torch.stack(labels)
    return res


train_ds_labeled = SPRTorchDataset(spr["train"], vocab, label2id, with_label=True)
dev_ds = SPRTorchDataset(spr["dev"], vocab, label2id, with_label=True)
train_ds_unlab = SPRTorchDataset(spr["train"], vocab, label2id, with_label=False)


# ---------- augmentations -------------
def mask_tokens(ids, mask_prob=0.15):
    ids = ids.clone()
    mask = (torch.rand_like(ids.float()) < mask_prob) & (ids != pad_id)
    ids[mask] = pad_id
    return ids


def shuffle_tokens(ids, win_size=4):
    ids = ids.clone()
    for row in ids:
        length = (row != pad_id).sum().item()
        tokens = row[:length]
        for start in range(0, length, win_size):
            end = min(start + win_size, length)
            segment = tokens[start:end]
            idx = torch.randperm(end - start)
            tokens[start:end] = segment[idx]
        row[:length] = tokens
    return ids


def augment(batch_ids):
    a1 = mask_tokens(batch_ids, 0.2)
    a1 = shuffle_tokens(a1, 4)
    a2 = mask_tokens(batch_ids, 0.2)
    a2 = shuffle_tokens(a2, 4)
    return a1, a2


# ---------- model ---------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, ids):
        emb = self.emb(ids)
        out, _ = self.lstm(emb)
        mask = (ids != pad_id).unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled  # (B,hid_dim)


class Classifier(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(
            (
                encoder.lstm.hidden_size * 2
                if hasattr(encoder.lstm, "hidden_size")
                else 256
            ),
            num_labels,
        )

    def forward(self, ids):
        return self.fc(self.encoder(ids))


# ---------- contrastive loss ----------
def nt_xent(z, temperature=0.5):
    z = nn.functional.normalize(z, dim=1)
    N = z.size(0) // 2
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    pos_idx = torch.arange(2 * N, device=z.device)
    pos_idx = pos_idx + N
    pos_idx = pos_idx % (2 * N)
    positives = sim[torch.arange(2 * N), pos_idx]
    denom = torch.logsumexp(sim, dim=1)
    loss = -positives + denom
    return loss.mean()


# ---------- pre-training --------------
def contrastive_pretrain(encoder, epochs=3, batch_size=256):
    loader = DataLoader(
        train_ds_unlab, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    optim = torch.optim.Adam(encoder.parameters(), lr=3e-4)
    for ep in range(1, epochs + 1):
        encoder.train()
        tot_loss = 0
        n = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            aug1, aug2 = augment(ids)
            aug1, aug2 = aug1.to(device), aug2.to(device)
            z1 = encoder(aug1)
            z2 = encoder(aug2)
            loss = nt_xent(torch.cat([z1, z2], 0))
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"Contrastive Epoch {ep}: loss={tot_loss/n:.4f}")


encoder = Encoder(len(vocab)).to(device)
contrastive_pretrain(encoder, epochs=3, batch_size=256)

# ---------- fine-tuning ---------------
model = Classifier(encoder, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    train_ds_labeled, batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

best_ccwa = -1
best_state = None
patience = 3
no_imp = 0
max_epochs = 15
for epoch in range(1, max_epochs + 1):
    # ---- train ----
    model.train()
    run_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["labels"].size(0)
    train_loss = run_loss / len(train_ds_labeled)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    # ---- eval ----
    model.eval()
    val_loss = 0
    all_pred = []
    all_true = []
    all_seq = []
    with torch.no_grad():
        for batch in dev_loader:
            tbatch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(tbatch["input_ids"])
            loss = criterion(logits, tbatch["labels"])
            val_loss += loss.item() * tbatch["labels"].size(0)
            preds = logits.argmax(-1).cpu().tolist()
            trues = tbatch["labels"].cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(trues)
            all_seq.extend(batch["sequences"])
    val_loss /= len(dev_ds)
    swa, cwa, ccwa = (
        SWA(all_seq, all_true, all_pred),
        CWA(all_seq, all_true, all_pred),
        CCWA(all_seq, all_true, all_pred),
    )
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"epoch": epoch, "swa": swa, "cwa": cwa, "ccwa": ccwa, "loss": val_loss}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} SWA={swa:.4f} CWA={cwa:.4f} CCWA={ccwa:.4f}"
    )
    # track best
    if ccwa > best_ccwa + 1e-5:
        best_ccwa = ccwa
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

# ---------- save data -----------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
