import os, pathlib, random, math, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import matthews_corrcoef, accuracy_score

# ---------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "RMA": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}
# ---------------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def maybe_dataset() -> DatasetDict:
    env_root = pathlib.Path(
        os.getenv("SPR_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    )
    if (env_root / "train.csv").exists():
        print("Loading real SPR_BENCH")
        return load_spr_bench(env_root)
    print("Real dataset not found → synthesising tiny dummy data")
    from datasets import Dataset as HFDS

    syms = list("ABCDEFGH")

    def synth(n):
        seq, lbl = [], []
        for i in range(n):
            ln = random.randint(5, 15)
            s = "".join(random.choice(syms) for _ in range(ln))
            seq.append(s)
            lbl.append(int(s.count("A") % 2 == 0))
        return HFDS.from_dict({"id": list(range(n)), "sequence": seq, "label": lbl})

    return DatasetDict(train=synth(2000), dev=synth(500), test=synth(500))


dsets = maybe_dataset()
print({k: len(v) for k, v in dsets.items()})

# ---------------------------------------------------------------------------
all_txt = "".join(dsets["train"]["sequence"])
vocab = sorted(set(all_txt))
stoi = {c: i + 1 for i, c in enumerate(vocab)}  # 0 pad
pad_idx = 0
max_len = min(60, max(len(s) for s in dsets["train"]["sequence"]))


def encode(seq):
    ids = [stoi.get(c, 0) for c in seq[:max_len]]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


class SPRTorch(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]
        self.labels = hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.float32),
            "rule": torch.tensor(len(self.seqs[idx]) % 5),
        }  # proxy rule-id


train_ds, val_ds, test_ds = (SPRTorch(dsets[s]) for s in ["train", "dev", "test"])


# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TinyTransformer(nn.Module):
    def __init__(self, vocab_sz, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz + 1, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, max_len)
        encoder = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(encoder, nlayers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        mask = x == pad_idx
        h = self.emb(x)
        h = self.pos(h)
        h = self.tr(h.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        h_masked = h.masked_fill(mask.unsqueeze(-1), 0.0)
        h_mean = h_masked.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.fc(h_mean).squeeze(1)


# ---------------------------------------------------------------------------
def rule_macro_accuracy(preds, labels, rules):
    # preds, labels: numpy arrays of 0/1; rules: int rule ids
    per_rule = {}
    for p, l, r in zip(preds, labels, rules):
        per_rule.setdefault(int(r), {"c": 0, "tot": 0})
        per_rule[int(r)]["c"] += int(p == l)
        per_rule[int(r)]["tot"] += 1
    accs = [v["c"] / v["tot"] for v in per_rule.values()]
    return np.mean(accs)


def run(lr, epochs=6, bs=128):
    model = TinyTransformer(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2)
    best_state = None
    best_mcc = -1
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot_loss = 0
        preds = []
        gts = []
        rules = []
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["x"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * batch["x"].size(0)
            preds += list((torch.sigmoid(logits).detach().cpu().numpy()) > 0.5)
            gts += list(batch["y"].cpu().numpy())
            rules += list(batch["rule"].cpu().numpy())
        train_loss = tot_loss / len(train_ds)
        train_mcc = matthews_corrcoef(gts, preds)
        train_rma = rule_macro_accuracy(preds, gts, rules)

        # ---- val ----
        model.eval()
        tot_loss = 0
        preds = []
        gts = []
        rules = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["x"])
                loss = crit(logits, batch["y"])
                tot_loss += loss.item() * batch["x"].size(0)
                preds += list((torch.sigmoid(logits).cpu().numpy()) > 0.5)
                gts += list(batch["y"].cpu().numpy())
                rules += list(batch["rule"].cpu().numpy())
        val_loss = tot_loss / len(val_ds)
        val_mcc = matthews_corrcoef(gts, preds)
        val_rma = rule_macro_accuracy(preds, gts, rules)

        experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train"].append(train_mcc)
        experiment_data["SPR_BENCH"]["metrics"]["val"].append(val_mcc)
        experiment_data["SPR_BENCH"]["RMA"]["train"].append(train_rma)
        experiment_data["SPR_BENCH"]["RMA"]["val"].append(val_rma)
        experiment_data["SPR_BENCH"]["epochs"].append(ep)

        print(
            f"lr {lr:.0e} | epoch {ep}: val_loss={val_loss:.4f}  MCC={val_mcc:.4f}  RMA={val_rma:.4f}"
        )
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_state = model.state_dict()
    return best_state, best_mcc


# ---------------------------------------------------------------------------
best_state = None
best_overall = -1
best_lr = None
for lr in [5e-4, 1e-3]:
    state, val_mcc = run(lr)
    if val_mcc > best_overall:
        best_overall = val_mcc
        best_state = state
        best_lr = lr
print(f"Best LR={best_lr:.0e} with dev MCC={best_overall:.4f}")

# ---------------------------------------------------------------------------
model_final = TinyTransformer(len(vocab)).to(device)
model_final.load_state_dict(best_state)
model_final.eval()
test_loader = DataLoader(test_ds, batch_size=256)
preds = []
gts = []
rules = []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = model_final(batch["x"])
        preds += list((torch.sigmoid(logits).cpu().numpy()) > 0.5)
        gts += list(batch["y"].cpu().numpy())
        rules += list(batch["rule"].cpu().numpy())
test_mcc = matthews_corrcoef(gts, preds)
test_rma = rule_macro_accuracy(preds, gts, rules)
print(f"Test MCC={test_mcc:.4f}, Rule-Macro Acc={test_rma:.4f}")

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["test_MCC"] = test_mcc
experiment_data["SPR_BENCH"]["test_RMA"] = test_rma
experiment_data["SPR_BENCH"]["best_lr"] = best_lr
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
