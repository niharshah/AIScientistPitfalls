import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment data container ----------------
experiment_data = {
    "hybrid_ctx_count": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------- data utilities ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


class SPRSeqDataset(Dataset):
    """provides (input_ids, counts, labels)"""

    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.max_len, self.pad = vocab, max_len, vocab["<pad>"]
        self.vocab_size = len(vocab)

    def _numericalise(self, s):
        return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in s[: self.max_len]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = self._numericalise(self.seqs[idx])
        pad_len = self.max_len - len(ids)
        ids_padded = ids + [self.pad] * pad_len
        # histogram counts excluding pad token
        cnt = np.bincount([i for i in ids if i != self.pad], minlength=self.vocab_size)
        return {
            "input_ids": torch.tensor(ids_padded, dtype=torch.long),
            "counts": torch.tensor(cnt, dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CausalLMDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs = hf_ds["sequence"]
        self.vocab, self.max_len, self.pad = vocab, max_len, vocab["<pad>"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [
            self.vocab.get(ch, self.vocab["<unk>"])
            for ch in self.seqs[idx][: self.max_len - 1]
        ]
        inp = [self.pad] + ids
        tgt = ids + [self.pad]
        pad_len = self.max_len - len(inp)
        inp += [self.pad] * pad_len
        tgt += [self.pad] * pad_len
        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(tgt, dtype=torch.long),
        }


# ---------------- model definitions ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout=dropout, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, causal=False):
        h = self.embed(x) + self.pos[:, : x.size(1)]
        if causal:
            L = x.size(1)
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h.transpose(0, 1), mask).transpose(0, 1)
        else:
            h = self.enc(h.transpose(0, 1)).transpose(0, 1)
        return h


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.encoder(x, causal=True)
        return self.lm_head(h)


class HybridSPRClassifier(nn.Module):
    """context transformer + bag-of-symbol statistics"""

    def __init__(self, vocab_size, num_classes, d_model, nhead, nlayer, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, nlayer, max_len)
        self.count_emb = nn.Parameter(torch.randn(vocab_size, d_model))
        self.dropout = nn.Dropout(0.2)
        self.cls_head = nn.Linear(2 * d_model, num_classes)

    def forward(self, x, counts):
        ctx_vec = self.encoder(x).mean(1)  # (B,d)
        cnt_vec = counts @ self.count_emb  # (B,d)
        vec = torch.cat([ctx_vec, cnt_vec], dim=1)
        vec = self.dropout(vec)
        return self.cls_head(vec)


# ---------------- training helpers ----------------
def train_causal_epoch(model, loader, optim, criterion):
    model.train()
    tot_loss, tot_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optim.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        loss.backward()
        optim.step()
        tot_loss += loss.item() * batch["labels"].numel()
        tot_tokens += batch["labels"].numel()
    return tot_loss / tot_tokens


def train_cls_epoch(model, loader, optim, criterion):
    model.train()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optim.zero_grad()
        out = model(batch["input_ids"], batch["counts"])
        loss = criterion(out, batch["labels"])
        loss.backward()
        optim.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    return tot_loss / len(loader.dataset), f1_score(gts, preds, average="macro")


@torch.no_grad()
def eval_cls_epoch(model, loader, criterion):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        out = model(batch["input_ids"], batch["counts"])
        loss = criterion(out, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------------- execution ----------------
def run():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)

    # vocabulary
    chars = set("".join(dsets["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({c: i + 2 for i, c in enumerate(sorted(chars))})
    vocab_size = len(vocab)
    max_len = min(128, max(len(s) for s in dsets["train"]["sequence"]) + 1)
    num_classes = len(set(dsets["train"]["label"]))
    print(f"Vocab={vocab_size}, Max_len={max_len}, Classes={num_classes}")

    # dataloaders
    pretrain_loader = DataLoader(
        CausalLMDataset(dsets["train"], vocab, max_len),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        SPRSeqDataset(dsets["train"], vocab, max_len), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(SPRSeqDataset(dsets["dev"], vocab, max_len), batch_size=256)
    test_loader = DataLoader(
        SPRSeqDataset(dsets["test"], vocab, max_len), batch_size=256
    )

    # model hyper-params
    d_model, nhead, nlayer = 128, 4, 3

    # -------- Stage 1: causal-LM pre-training ----------
    lm = CausalLM(vocab_size, d_model, nhead, nlayer, max_len).to(device)
    opt_lm = torch.optim.Adam(lm.parameters(), lr=1e-3)
    crit_lm = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    for ep in range(1, 4):
        loss = train_causal_epoch(lm, pretrain_loader, opt_lm, crit_lm)
        experiment_data["hybrid_ctx_count"]["losses"]["pretrain"].append(loss)
        print(f"Pre-train epoch {ep}: loss={loss:.4f}")

    encoder_state = lm.encoder.state_dict()

    # -------- Stage 2: hybrid classifier fine-tune ----------
    model = HybridSPRClassifier(
        vocab_size, num_classes, d_model, nhead, nlayer, max_len
    ).to(device)
    model.encoder.load_state_dict(encoder_state)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    best_f1, patience, wait, best_state = 0.0, 5, 0, None
    for ep in range(1, 21):
        tr_loss, tr_f1 = train_cls_epoch(model, train_loader, opt, crit)
        val_loss, val_f1, _, _ = eval_cls_epoch(model, val_loader, crit)

        ed = experiment_data["hybrid_ctx_count"]
        ed["epochs"].append(ep)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_macro_f1"].append(tr_f1)
        ed["metrics"]["val_macro_f1"].append(val_f1)

        print(f"Epoch {ep}: val_loss={val_loss:.4f}, val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

    # -------- Test evaluation ----------
    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = eval_cls_epoch(model, test_loader, crit)
    print(f"TEST macro-F1 = {test_f1:.4f}")

    ed = experiment_data["hybrid_ctx_count"]
    ed["predictions"], ed["ground_truth"] = preds, gts
    ed["test_macro_f1"], ed["test_loss"] = test_f1, test_loss

    # save all
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


run()
