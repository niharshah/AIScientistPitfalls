import os, random, string, datetime, json, math, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------- house-keeping -------------------------------------------------
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

work_dir = os.path.join(os.getcwd(), "working")
os.makedirs(work_dir, exist_ok=True)

experiment_data = {"dropout_rate": {"SPR_BENCH": {}}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -------------------- data (real or synthetic) ----------------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(p):
    return all(
        os.path.isfile(os.path.join(p, f"{s}.csv")) for s in ["train", "dev", "test"]
    )


use_synth = not spr_files_exist(SPR_PATH)

if use_synth:
    print("Generating synthetic SPR-like dataset.")
    shapes = list(string.ascii_uppercase[:6])
    colors = list(map(str, range(4)))

    def rnd_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(seq):
        us = len(set(t[0] for t in seq.split()))
        uc = len(set(t[1] for t in seq.split()))
        return int(us == uc)

    def make(n):
        seq = [rnd_seq() for _ in range(n)]
        return {"sequence": seq, "label": [rule(s) for s in seq]}

    raw = {"train": make(2000), "dev": make(400), "test": make(600)}
else:
    from datasets import load_dataset, DatasetDict

    def load_spr(root):
        def l(split):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, f"{split}.csv"),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict()
        d["train"] = l("train")
        d["dev"] = l("dev")
        d["test"] = l("test")
        return d

    ds = load_spr(SPR_PATH)
    raw = {
        split: {"sequence": ds[split]["sequence"], "label": ds[split]["label"]}
        for split in ["train", "dev", "test"]
    }

# ---------------------- helpers -------------------------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    toks = {tok for s in seqs for tok in s.split()}
    vocab = {PAD: 0, UNK: 1}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw["train"]["sequence"])
V = len(vocab)
print("Vocab size:", V)


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


def count_shape_variety(s):
    return len(set(t[0] for t in s.split()))


def count_color_variety(s):
    return len(set(t[1] for t in s.split()))


def swa(seq, y, p):
    w = [count_shape_variety(s) for s in seq]
    return sum(wi for wi, t, r in zip(w, y, p) if t == r) / (sum(w) or 1)


def cwa(seq, y, p):
    w = [count_color_variety(s) for s in seq]
    return sum(wi for wi, t, r in zip(w, y, p) if t == r) / (sum(w) or 1)


def signatures(seqs):
    out = []
    for s in seqs:
        out.append(
            (
                tuple(sorted(set(t[0] for t in s.split()))),
                tuple(sorted(set(t[1] for t in s.split()))),
            )
        )
    return out


# ----------------------- torch Dataset -----------------------------------------
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw = seqs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.X[i], "label": self.y[i]}


def collate(batch):
    lens = [len(b["input_ids"]) for b in batch]
    maxlen = max(lens)
    ids = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    y = torch.empty(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : lens[i]] = b["input_ids"]
        y[i] = b["label"]
    return {"input_ids": ids, "labels": y, "lengths": torch.tensor(lens)}


datasets = {
    s: SPRDataset(raw[s]["sequence"], raw[s]["label"]) for s in ["train", "dev", "test"]
}


# ----------------------- model --------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_cls, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab[PAD])
        self.drop_emb = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.drop_h = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, num_cls)

    def forward(self, x, lens):
        x = self.drop_emb(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = self.drop_h(h.squeeze(0))
        return self.out(h)


# ----------------------- training utilities ------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            logits = model(b["input_ids"], b["lengths"])
            loss = criterion(logits, b["labels"])
            loss_sum += loss.item() * b["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == b["labels"]).sum().item()
            tot += b["labels"].size(0)
    return loss_sum / tot, correct / tot


def full_preds(model, seqs, bsize=64):
    pred = []
    with torch.no_grad():
        for i in range(0, len(seqs), bsize):
            enc = [encode(s) for s in seqs[i : i + bsize]]
            lens = torch.tensor([len(e) for e in enc])
            maxlen = lens.max()
            x = torch.full((len(enc), maxlen), vocab[PAD], dtype=torch.long)
            for j, row in enumerate(enc):
                x[j, : len(row)] = torch.tensor(row)
            logits = model(x.to(device), lens.to(device))
            pred.extend(logits.argmax(-1).cpu().tolist())
    return pred


# ----------------------- hyper-parameter loop ----------------------------------
drop_rates = [0.0, 0.2, 0.3, 0.5]
batch_size = 64
epochs = 6
for rate in drop_rates:
    tag = str(rate)
    exp_entry = {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "NRGS": [],
        "timestamps": [],
    }
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "dev", "test"]
    }
    model = GRUClassifier(V, 64, 128, len(set(raw["train"]["label"])), rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # training
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0
        for b in loaders["train"]:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            logits = model(b["input_ids"], b["lengths"])
            loss = criterion(logits, b["labels"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            run_loss += loss.item() * b["labels"].size(0)
        tr_loss = run_loss / len(datasets["train"])
        val_loss, val_acc = evaluate(model, loaders["dev"], criterion)
        exp_entry["losses"]["train"].append(tr_loss)
        exp_entry["losses"]["val"].append(val_loss)
        exp_entry["metrics"]["train"].append(tr_loss)  # storing simple metric
        exp_entry["metrics"]["val"].append(val_acc)
        exp_entry["timestamps"].append(str(datetime.datetime.now()))
        print(
            f"dropout={rate}  epoch={ep}  train_loss={tr_loss:.3f}  val_loss={val_loss:.3f}  val_acc={val_acc:.3f}"
        )
    # test & NRGS
    _, test_acc = evaluate(model, loaders["test"], criterion)
    seqs = datasets["test"].raw
    gts = datasets["test"].y.tolist()
    preds = full_preds(model, seqs)
    swa_val = swa(seqs, gts, preds)
    cwa_val = cwa(seqs, gts, preds)
    train_sigs = set(signatures(raw["train"]["sequence"]))
    novel = [i for i, sg in enumerate(signatures(seqs)) if sg not in train_sigs]
    NRGS = (sum(1 for i in novel if preds[i] == gts[i]) / len(novel)) if novel else 0.0
    exp_entry["metrics"]["test"] = {"acc": test_acc, "swa": swa_val, "cwa": cwa_val}
    exp_entry["predictions"] = preds
    exp_entry["ground_truth"] = gts
    exp_entry["NRGS"] = NRGS
    experiment_data["dropout_rate"]["SPR_BENCH"][tag] = exp_entry
    # save simple loss curve for last epoch of this rate
    plt.figure()
    plt.plot(exp_entry["losses"]["train"], label="train")
    plt.plot(exp_entry["losses"]["val"], label="val")
    plt.title(f"Loss (dropout={rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(work_dir, f"loss_curve_dropout_{rate}.png"))
    plt.close()
    print(
        f"TEST  acc={test_acc:.3f}  SWA={swa_val:.3f}  CWA={cwa_val:.3f}  NRGS={NRGS:.3f}\n"
    )

# ----------------------- persist ------------------------------------------------
np.save(
    os.path.join(work_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
with open(os.path.join(work_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
print("Finished; data saved.")
