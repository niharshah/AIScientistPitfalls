import os, math, pathlib, random, time, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------------- working dir & device ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment data container ----------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "SGA": [], "test_f1": None},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------------- dataset utils ----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


for p in [
    pathlib.Path("./SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("SPR_BENCH not found")

spr = load_spr_bench(DATA_PATH)

PAD, UNK = "<PAD>", "<UNK>"
vocab = [PAD, UNK] + sorted({c for s in spr["train"]["sequence"] for c in s})
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
VOCAB, N_CLASS = len(vocab), len(label2id)
MAX_LEN = 64
N_BIGRAM = 256  # hashed bag-of-bigrams size


def encode_seq(seq):
    ids = [stoi.get(c, stoi[UNK]) for c in seq[:MAX_LEN]]
    ids += [stoi[PAD]] * (MAX_LEN - len(ids))
    return ids


def uni_vec(seq):
    v = np.zeros(VOCAB, dtype=np.float32)
    for c in seq:
        v[stoi.get(c, 1)] += 1
    if v.sum():
        v /= v.sum()
    return v


def bi_vec(seq):
    v = np.zeros(N_BIGRAM, dtype=np.float32)
    for i in range(len(seq) - 1):
        idx = hash(seq[i : i + 2]) % N_BIGRAM
        v[idx] += 1
    if v.sum():
        v /= v.sum()
    return v


class SPRTorch(Dataset):
    def __init__(self, split):
        self.seqs, self.labs = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode_seq(s), dtype=torch.long),
            "sym_feats": torch.tensor(
                np.concatenate(
                    [
                        uni_vec(s),
                        bi_vec(s),
                        np.array([len(s) / MAX_LEN], dtype=np.float32),
                    ]
                ),
                dtype=torch.float32,
            ),
            "labels": torch.tensor(label2id[self.labs[idx]], dtype=torch.long),
        }


BATCH = 128
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=BATCH, shuffle=True)
val_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=BATCH)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=BATCH)


# ---------------- model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN + 1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.embed = nn.Embedding(VOCAB, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, 8, 512, 0.1, batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer, 4)
        sym_in = VOCAB + N_BIGRAM + 1
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_in, 256), nn.ReLU(), nn.Linear(256, d_model)
        )
        self.out = nn.Linear(d_model * 2, N_CLASS)

    def forward(self, ids, feats):
        B = ids.size(0)
        tok = self.cls.expand(B, -1, -1)
        x = torch.cat([tok, self.embed(ids)], 1)
        x = self.trans(
            self.pos(x),
            src_key_padding_mask=torch.cat(
                [torch.zeros(B, 1, device=ids.device).bool(), ids.eq(0)], 1
            ),
        )
        seq_emb = x[:, 0]  # CLS
        sym_emb = torch.relu(self.sym_mlp(feats))
        return self.out(torch.cat([seq_emb, sym_emb], 1))


model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()


# ---------------- train / eval helpers ----------------
def epoch_run(dl, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, labs = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(train):
            out = model(batch["input_ids"], batch["sym_feats"])
            loss = crit(out, batch["labels"])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.append(out.argmax(-1).cpu())
        labs.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labs = torch.cat(labs).numpy()
    return (
        tot_loss / len(dl.dataset),
        f1_score(labs, preds, average="macro"),
        preds,
        labs,
    )


def bigram_set(s):
    return {s[i : i + 2] for i in range(len(s) - 1)}


train_bigrams = set().union(*(bigram_set(s) for s in spr["train"]["sequence"]))


def compute_SGA(preds):
    ood = np.array(
        [len(bigram_set(s) - train_bigrams) > 0 for s in spr["test"]["sequence"]]
    )
    correct = preds == np.array([label2id[l] for l in spr["test"]["label"]])
    return correct[ood].mean() if ood.any() else 0.0


# ---------------- training loop ----------------
BEST, wait, PAT = -1, 0, 4
for ep in range(1, 31):
    tr_loss, tr_f1, _, _ = epoch_run(train_dl, True)
    vl_loss, vl_f1, _, _ = epoch_run(val_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(vl_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(vl_f1)
    experiment_data["SPR_BENCH"]["epochs"].append(ep)
    print(f"Epoch {ep}: val_loss={vl_loss:.4f} val_F1={vl_f1:.4f}")
    if vl_f1 > BEST:
        BEST = vl_f1
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
    if wait >= PAT:
        print("Early stopping")
        break

model.load_state_dict(best_state)

# ---------------- final test ----------------
ts_loss, ts_f1, ts_preds, ts_labels = epoch_run(test_dl)
SGA = compute_SGA(ts_preds)
print(f"Test macro-F1={ts_f1:.4f} | SGA={SGA:.4f}")

experiment_data["SPR_BENCH"]["test_f1"] = ts_f1
experiment_data["SPR_BENCH"]["metrics"]["SGA"].append(SGA)
experiment_data["SPR_BENCH"]["predictions"] = ts_preds.tolist()
experiment_data["SPR_BENCH"]["ground_truth"] = ts_labels.tolist()

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
