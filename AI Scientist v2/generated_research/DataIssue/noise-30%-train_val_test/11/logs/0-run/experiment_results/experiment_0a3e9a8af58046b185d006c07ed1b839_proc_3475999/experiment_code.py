import os, math, random, time, pathlib, csv, numpy as np

# --------------------------- working dir ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# --------------------------- reproducibility ---------------------------
def set_seed(seed: int = 42):
    import torch, os, numpy as np, random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# --------------------------- torch device ---------------------------
import torch, torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------- small synthetic benchmark ---------------------------
TOKENS = [chr(i) for i in range(65, 91)]  # A-Z


def gen_example(rule: str, idx: int):
    length = random.randint(5, 12)
    seq_tokens = random.choices(TOKENS, k=length)
    if rule == "evenA":
        label = "evenA" if seq_tokens.count("A") % 2 == 0 else "oddA"
    elif rule == "majority_vowel":
        vowels = {"A", "E", "I", "O", "U"}
        label = (
            "majVowel"
            if sum(t in vowels for t in seq_tokens) > len(seq_tokens) / 2
            else "majCons"
        )
    elif rule == "has_sub_AB":
        label = (
            "containsAB"
            if any(a == "A" and b == "B" for a, b in zip(seq_tokens, seq_tokens[1:]))
            else "noAB"
        )
    else:
        raise ValueError(rule)
    return (idx, " ".join(seq_tokens), label)


def build_dataset(root: pathlib.Path, rule: str, n_train=2000, n_dev=500, n_test=700):
    root.mkdir(parents=True, exist_ok=True)

    def dump(split, n):
        with (root / f"{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", "label"])
            for i in range(n):
                w.writerow(gen_example(rule, i))

    dump("train", n_train)
    dump("dev", n_dev)
    dump("test", n_test)


from datasets import load_dataset, DatasetDict


def load_csv_dataset(root):
    return DatasetDict(
        {
            sp: load_dataset(
                "csv",
                data_files=str(root / f"{sp}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )
            for sp in ["train", "dev", "test"]
        }
    )


# --------------------------- helpers ---------------------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(seq, vocab, max_len):
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


from torch.utils.data import Dataset, DataLoader


class TxtDataset(Dataset):
    def __init__(self, split, vocab, max_len, label2id):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx], self.vocab, self.max_len)),
            "labels": torch.tensor(self.labels[idx]),
        }


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ReasoningTransformer(nn.Module):
    def __init__(self, vocab, d_model, heads, layers, num_labels, drop=0.1, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pe = PosEnc(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, heads, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.proj_rel = nn.Linear(d_model, d_model, bias=False)
        self.cls = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d_model, num_labels),
        )

    def forward(self, input_ids):
        mask = input_ids == 0
        x = self.pe(self.emb(input_ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        x_mask = x.masked_fill(mask.unsqueeze(-1), 0)
        pooled = x_mask.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        rel_scores = torch.relu(
            torch.matmul(self.proj_rel(x), x.transpose(1, 2))
        ).softmax(-1)
        rel_vec = torch.bmm(rel_scores, x).mean(1)
        return self.cls(torch.cat([pooled, rel_vec], -1))


def metrics_from_preds(trues, preds):
    from sklearn.metrics import f1_score, accuracy_score

    return f1_score(trues, preds, average="macro"), accuracy_score(trues, preds)


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        if train:
            opt.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            if train:
                loss.backward()
                opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    macro_f1, ema = metrics_from_preds(trues, preds)
    return tot_loss / len(loader.dataset), macro_f1, ema, preds, trues


# --------------------------- main experiment loop ---------------------------
experiment_data = {}
DATASETS = ["evenA", "majority_vowel", "has_sub_AB"]
for ds in DATASETS:
    root = pathlib.Path(f"./SPR_{ds}")
    if not root.exists():
        build_dataset(root, ds)
    dset = load_csv_dataset(root)
    max_len = min(max(len(s.split()) for s in dset["train"]["sequence"]), 64)
    vocab = build_vocab(dset["train"]["sequence"])
    labels = sorted(set(dset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    loaders = {
        sp: DataLoader(
            TxtDataset(dset[sp], vocab, max_len, label2id),
            batch_size=64,
            shuffle=(sp == "train"),
        )
        for sp in ["train", "dev", "test"]
    }
    # --- class weights bug-fix ---
    counts = np.bincount(
        [label2id[l] for l in dset["train"]["label"]], minlength=len(labels)
    )
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(labels)
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    # --- model, loss, optimiser ---
    set_seed(42)
    model = ReasoningTransformer(len(vocab), 128, 4, 2, len(labels), 0.1, max_len).to(
        device
    )
    crit = nn.CrossEntropyLoss(weight=weights_t)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    rec = {
        "metrics": {"train_f1": [], "val_f1": [], "train_ema": [], "val_ema": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    # --- training ---
    for epoch in range(1, 6):
        tr_loss, tr_f1, tr_ema, _, _ = run_epoch(model, loaders["train"], crit, opt)
        vl_loss, vl_f1, vl_ema, _, _ = run_epoch(model, loaders["dev"], crit, None)
        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(vl_loss)
        rec["metrics"]["train_f1"].append(tr_f1)
        rec["metrics"]["val_f1"].append(vl_f1)
        rec["metrics"]["train_ema"].append(tr_ema)
        rec["metrics"]["val_ema"].append(vl_ema)
        rec["epochs"].append(epoch)
        print(
            f"[{ds}] Epoch {epoch}: val_loss={vl_loss:.4f} | val_F1={vl_f1:.4f} | val_EMA={vl_ema:.4f}"
        )
    # --- test ---
    ts_loss, ts_f1, ts_ema, preds, trues = run_epoch(model, loaders["test"], crit, None)
    rec["test_loss"] = ts_loss
    rec["test_f1"] = ts_f1
    rec["test_ema"] = ts_ema
    rec["predictions"] = preds
    rec["ground_truth"] = trues
    experiment_data[ds] = rec
    print(f"[{ds}] Test: loss={ts_loss:.4f} | macroF1={ts_f1:.4f} | EMA={ts_ema:.4f}")

# --------------------------- save ---------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Saved all metrics to {os.path.join(working_dir, 'experiment_data.npy')}")
