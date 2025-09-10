import os, math, random, time, pathlib, csv

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- imports ---------------------------
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict
from typing import List, Dict

# --------------------------- device ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- reproducibility ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# --------------------------- synthetic dataset generation ---------------------------
TOKENS = [chr(i) for i in range(65, 91)]  # 'A'-'Z'


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


# --------------------------- vocab & dataset ---------------------------
PAD, UNK = "<pad>", "<unk>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for s in seqs:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode(seq: str, vocab, max_len):
    ids = [vocab.get(tok, vocab[UNK]) for tok in seq.split()][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class TxtDataset(Dataset):
    def __init__(self, split, vocab, max_len, label2id):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx], self.vocab, self.max_len)),
            "labels": torch.tensor(self.labels[idx]),
        }


# --------------------------- model ---------------------------
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
    def __init__(
        self, vocab_size, emb, heads, layers, num_labels, drop=0.1, max_len=64
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.pe = PosEnc(emb, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.proj_rel = nn.Linear(emb, emb, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(emb * 2, emb),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(emb, num_labels),
        )

    def forward(self, input_ids):
        mask = input_ids == 0  # [B,L]
        x = self.emb(input_ids)
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        # standard mean pooling
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
        pooled = x_masked.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        # relation module with proper padding mask
        rel_logits = torch.matmul(self.proj_rel(x), x.transpose(1, 2)) / math.sqrt(
            x.size(-1)
        )
        rel_logits = rel_logits.masked_fill(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(rel_logits, dim=-1)
        rel_vec = torch.bmm(attn, x).mean(1)
        logits = self.classifier(torch.cat([pooled, rel_vec], -1))
        return logits


# --------------------------- training helpers ---------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, trues = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            if train:
                loss.backward()
                optimizer.step()
        bs = batch["labels"].size(0)
        tot_loss += loss.item() * bs
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    avg_loss = tot_loss / len(loader.dataset)
    macro_f1 = f1_score(trues, preds, average="macro")
    ema = accuracy_score(trues, preds)
    return avg_loss, macro_f1, ema, preds, trues


# --------------------------- experiment loop ---------------------------
experiment_data = {}
DATASETS = ["evenA", "majority_vowel", "has_sub_AB"]
for ds_name in DATASETS:
    print(f"\n=== Dataset: {ds_name} ===")
    root = pathlib.Path(f"./SPR_{ds_name}")
    if not root.exists():
        build_dataset(root, ds_name)
    dset = load_csv_dataset(root)
    max_len = min(max(len(s.split()) for s in dset["train"]["sequence"]), 64)
    vocab = build_vocab(dset["train"]["sequence"])
    label_set = sorted(set(dset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(label_set)}

    loaders = {
        sp: DataLoader(
            TxtDataset(dset[sp], vocab, max_len, label2id),
            batch_size=64,
            shuffle=(sp == "train"),
        )
        for sp in ["train", "dev", "test"]
    }

    model = ReasoningTransformer(len(vocab), 128, 4, 2, len(label2id), 0.1, max_len).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    rec = {
        "metrics": {
            "train_macro_f1": [],
            "val_macro_f1": [],
            "train_ema": [],
            "val_ema": [],
        },
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": None,
        "ground_truth": None,
    }

    epochs = 5
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, tr_ema, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer
        )
        vl_loss, vl_f1, vl_ema, _, _ = run_epoch(model, loaders["dev"], criterion)
        rec["losses"]["train"].append(tr_loss)
        rec["losses"]["val"].append(vl_loss)
        rec["metrics"]["train_macro_f1"].append(tr_f1)
        rec["metrics"]["val_macro_f1"].append(vl_f1)
        rec["metrics"]["train_ema"].append(tr_ema)
        rec["metrics"]["val_ema"].append(vl_ema)
        rec["epochs"].append(epoch)
        print(
            f"Epoch {epoch}: val_loss={vl_loss:.4f}  valF1={vl_f1:.4f}  valEMA={vl_ema:.4f}"
        )

    ts_loss, ts_f1, ts_ema, preds, trues = run_epoch(model, loaders["test"], criterion)
    rec["test_loss"] = ts_loss
    rec["test_macro_f1"] = ts_f1
    rec["test_ema"] = ts_ema
    rec["predictions"] = np.array(preds)
    rec["ground_truth"] = np.array(trues)

    experiment_data[ds_name] = rec
    print(f"[{ds_name}] Test F1={ts_f1:.4f}  Test EMA={ts_ema:.4f}")

# --------------------------- save results ---------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
