import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "hybrid_ctx+count": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- dataset helpers ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


class SPRSeqSet(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]
        self.vocab, self.max_len = vocab, max_len
        self.pad = vocab["<pad>"]
        self.unk = vocab["<unk>"]

    def __len__(self):
        return len(self.seqs)

    def _encode(self, s):
        ids = [self.vocab.get(ch, self.unk) for ch in s[: self.max_len]]
        ids += [self.pad] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self._encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CausalLMSet(Dataset):
    def __init__(self, hf_ds, vocab, max_len):
        self.seqs, self.vocab, self.max_len = hf_ds["sequence"], vocab, max_len
        self.pad, self.unk = vocab["<pad>"], vocab["<unk>"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_ids = [
            self.vocab.get(ch, self.unk) for ch in self.seqs[idx][: self.max_len - 1]
        ]
        inp = [self.pad] + seq_ids
        tgt = seq_ids + [self.pad]
        inp += [self.pad] * (self.max_len - len(inp))
        tgt += [self.pad] * (self.max_len - len(tgt))
        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(tgt, dtype=torch.long),
        }


# ---------- model blocks ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, max_len, drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=drop, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)

    def forward(self, x, causal=False):
        h = self.embed(x) + self.pos[:, : x.size(1)]
        if causal:
            L = x.size(1)
            mask = torch.triu(torch.ones(L, L, device=x.device).bool(), 1)
            h = self.enc(h.transpose(0, 1), mask).transpose(0, 1)
        else:
            h = self.enc(h.transpose(0, 1)).transpose(0, 1)
        return h  # (B,L,D)


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, nlayers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # (B,L)
        h = self.encoder(x, causal=True)
        return self.lm_head(h)


class HybridClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_classes,
        d_model,
        nhead,
        nlayers,
        max_len,
        count_dim=64,
        drop=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, nlayers, max_len, drop)
        self.count_proj = nn.Sequential(
            nn.LayerNorm(vocab_size), nn.Linear(vocab_size, count_dim), nn.ReLU()
        )
        self.class_head = nn.Linear(d_model + count_dim, n_classes)
        self.pad_id = 0

    def forward(self, x):  # x (B,L)
        B, L = x.size()
        enc = self.encoder(x)  # (B,L,D)
        ctx_vec = enc.mean(1)  # (B,D)
        # token counts (exclude pad)
        one_hot = torch.nn.functional.one_hot(
            x, num_classes=self.count_proj[1].in_features
        ).float()
        one_hot = one_hot.masked_fill((x == self.pad_id).unsqueeze(-1), 0.0)
        counts = one_hot.sum(1)  # (B,V)
        count_feat = self.count_proj(counts)
        feat = torch.cat([ctx_vec, count_feat], dim=-1)
        return self.class_head(feat)


# ---------- train / eval helpers ----------
def train_lm_epoch(model, loader, opt, crit):
    model.train()
    tot = 0
    ls = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = crit(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        loss.backward()
        opt.step()
        ls += loss.item() * batch["labels"].numel()
        tot += batch["labels"].numel()
    return ls / tot


def cls_run_epoch(model, loader, opt, crit, train=True):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = 0
    gts = []
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(batch["input_ids"])
            loss = crit(out, batch["labels"])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), f1, preds, gts


# ---------- main experiment ----------
def run():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)

    # vocab & lengths
    chars = set("".join(dsets["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({c: i + 2 for i, c in enumerate(sorted(chars))})
    max_len = min(128, max(len(s) for s in dsets["train"]["sequence"]) + 1)
    V = len(vocab)
    n_cls = len(set(dsets["train"]["label"]))
    print(f"Vocab {V}, classes {n_cls}, max_len {max_len}")

    # loaders
    bs_lm, bs_cls = 256, 128
    train_lm_loader = DataLoader(
        CausalLMSet(dsets["train"], vocab, max_len),
        batch_size=bs_lm,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        SPRSeqSet(dsets["train"], vocab, max_len), batch_size=bs_cls, shuffle=True
    )
    val_loader = DataLoader(SPRSeqSet(dsets["dev"], vocab, max_len), batch_size=256)
    test_loader = DataLoader(SPRSeqSet(dsets["test"], vocab, max_len), batch_size=256)

    # model sizes
    d_model, nhead, nlayers = 128, 4, 3

    # ---------- stage1: LM pretrain ----------
    lm = CausalLM(V, d_model, nhead, nlayers, max_len).to(device)
    opt_lm = torch.optim.Adam(lm.parameters(), lr=1e-3)
    crit_lm = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    for ep in range(3):
        l = train_lm_epoch(lm, train_lm_loader, opt_lm, crit_lm)
        experiment_data["hybrid_ctx+count"]["losses"]["pretrain"].append(l)
        print(f"[LM] epoch {ep+1}: loss {l:.4f}")

    encoder_state = lm.encoder.state_dict()

    # ---------- stage2: hybrid classification ----------
    model = HybridClassifier(V, n_cls, d_model, nhead, nlayers, max_len).to(device)
    model.encoder.load_state_dict(encoder_state)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    crit_cls = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1, patience, wait, best_state = 0.0, 5, 0, None
    max_epochs = 15
    for ep in range(1, max_epochs + 1):
        tr_loss, tr_f1, _, _ = cls_run_epoch(model, train_loader, opt, crit_cls, True)
        val_loss, val_f1, _, _ = cls_run_epoch(model, val_loader, opt, crit_cls, False)
        scheduler.step()

        ed = experiment_data["hybrid_ctx+count"]
        ed["epochs"].append(ep)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_macro_f1"].append(tr_f1)
        ed["metrics"]["val_macro_f1"].append(val_f1)

        print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stop")
            break

    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = cls_run_epoch(
        model, test_loader, opt, crit_cls, False
    )
    print(f"TEST Macro-F1 = {test_f1:.4f}")

    ed = experiment_data["hybrid_ctx+count"]
    ed["predictions"], ed["ground_truth"] = preds, gts
    ed["test_macro_f1"], ed["test_loss"] = test_f1, test_loss

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


run()
