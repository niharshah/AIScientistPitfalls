import os, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -------------------- working dir --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- experiment data container --------------------
experiment_data = {
    "dual_stream": {
        "losses": {"pretrain": [], "train": [], "val": []},
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# -------------------- dataset helpers --------------------
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


class SPRDualDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len, with_counts=True, shift_for_lm=False):
        self.seq = hf_ds["sequence"]
        self.labels = hf_ds["label"] if "label" in hf_ds.column_names else None
        self.vocab, self.max_len = vocab, max_len
        self.pad = vocab["<pad>"]
        self.with_counts = with_counts
        self.shift_for_lm = shift_for_lm  # for causal LM pre-training

    def __len__(self):
        return len(self.seq)

    def encode(self, s):
        ids = [self.vocab.get(ch, self.vocab["<unk>"]) for ch in s[: self.max_len]]
        return ids

    def __getitem__(self, idx):
        ids = self.encode(self.seq[idx])
        if self.shift_for_lm:  # LM data (predict next)
            inp = [self.pad] + ids
            tgt = ids + [self.pad]
            inp = inp[: self.max_len]
            tgt = tgt[: self.max_len]
            inp += [self.pad] * (self.max_len - len(inp))
            tgt += [self.pad] * (self.max_len - len(tgt))
            return {
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "labels": torch.tensor(tgt, dtype=torch.long),
            }
        # classification data
        padded = ids + [self.pad] * (self.max_len - len(ids))
        sample = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.with_counts:
            cnt = torch.bincount(
                torch.tensor(ids, dtype=torch.long), minlength=len(self.vocab)
            ).float()
            sample["counts"] = cnt
        return sample


# -------------------- model definitions --------------------
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, max_len, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoder(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, x, causal=False):
        h = self.tok_emb(x) + self.pos_enc(x)
        if causal:
            L = x.size(1)
            mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )
            h = self.transformer(h, mask)
        else:
            h = self.transformer(h)
        return h


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, max_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, nlayers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.encoder(x, causal=True)
        return self.lm_head(h)


class DualStreamClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_classes,
        d_model,
        nhead,
        nlayers,
        max_len,
        count_dim=64,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, nlayers, max_len)
        self.count_mlp = nn.Sequential(
            nn.Linear(vocab_size, count_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(d_model + count_dim, n_classes)

    def forward(self, x, counts):
        enc = self.encoder(x).mean(1)  # (B, d_model)
        cnt_feat = self.count_mlp(counts)  # (B, count_dim)
        feat = torch.cat([enc, cnt_feat], dim=-1)
        return self.head(feat)


# -------------------- training utils --------------------
def train_lm_epoch(model, loader, opt, crit):
    model.train()
    tot, loss_sum = 0, 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        logits = model(batch["input_ids"])
        loss = crit(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        loss.backward()
        opt.step()
        tot += batch["labels"].numel()
        loss_sum += loss.item() * batch["labels"].numel()
    return loss_sum / tot


def train_cls_epoch(model, loader, opt, crit):
    model.train()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = model(batch["input_ids"], batch["counts"])
        loss = crit(out, batch["labels"])
        loss.backward()
        opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), f1


@torch.no_grad()
def eval_cls_epoch(model, loader, crit):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["counts"])
        loss = crit(out, batch["labels"])
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(batch["labels"].cpu().numpy())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(loader.dataset), f1, preds, gts


# -------------------- main routine --------------------
def run():
    DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    dsets = load_spr_bench(DATA_PATH)

    # -------- vocabulary -------------
    chars = set("".join(dsets["train"]["sequence"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({c: i + 2 for i, c in enumerate(sorted(chars))})
    vocab_size = len(vocab)
    max_len = min(128, max(len(s) for s in dsets["train"]["sequence"]) + 1)
    print("Vocab", vocab_size, "Max_len", max_len)

    # -------- data loaders ----------
    pretrain_loader = DataLoader(
        SPRDualDataset(
            dsets["train"], vocab, max_len, with_counts=False, shift_for_lm=True
        ),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        SPRDualDataset(dsets["train"], vocab, max_len), batch_size=128, shuffle=True
    )

    val_loader = DataLoader(
        SPRDualDataset(dsets["dev"], vocab, max_len), batch_size=256
    )

    test_loader = DataLoader(
        SPRDualDataset(dsets["test"], vocab, max_len), batch_size=256
    )

    n_classes = len(set(dsets["train"]["label"]))
    d_model, nhead, nlayers = 128, 4, 3

    # -------- stage 1 : pre-train ----------
    lm = CausalLM(vocab_size, d_model, nhead, nlayers, max_len).to(device)
    opt_lm = torch.optim.Adam(lm.parameters(), lr=1e-3)
    crit_lm = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    for ep in range(1, 3):
        ploss = train_lm_epoch(lm, pretrain_loader, opt_lm, crit_lm)
        experiment_data["dual_stream"]["losses"]["pretrain"].append(ploss)
        print(f"Pretrain {ep}/2 : loss={ploss:.4f}")

    # keep encoder weights
    enc_state = lm.encoder.state_dict()

    # -------- stage 2 : fine-tune classifier ----------
    model = DualStreamClassifier(
        vocab_size, n_classes, d_model, nhead, nlayers, max_len
    ).to(device)
    model.encoder.load_state_dict(enc_state)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    patience, best_f1, wait = 5, 0.0, 0
    best_state = None

    for ep in range(1, 13):
        tr_loss, tr_f1 = train_cls_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, _, _ = eval_cls_epoch(model, val_loader, criterion)
        scheduler.step()

        ed = experiment_data["dual_stream"]
        ed["epochs"].append(ep)
        ed["losses"]["train"].append(tr_loss)
        ed["losses"]["val"].append(val_loss)
        ed["metrics"]["train_macro_f1"].append(tr_f1)
        ed["metrics"]["val_macro_f1"].append(val_f1)

        print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, wait, best_state = val_f1, 0, model.state_dict()
        else:
            wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

    # -------- test evaluation ----------
    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_f1, preds, gts = eval_cls_epoch(model, test_loader, criterion)
    print(f"TEST Macro-F1 = {test_f1:.4f}")

    ed = experiment_data["dual_stream"]
    ed["predictions"], ed["ground_truth"] = preds, gts
    ed["test_macro_f1"], ed["test_loss"] = test_f1, test_loss

    # save
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


run()
