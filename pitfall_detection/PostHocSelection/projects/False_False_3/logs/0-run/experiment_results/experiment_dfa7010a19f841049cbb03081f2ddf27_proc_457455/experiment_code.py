import os, pathlib, random, numpy as np, torch
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict

# ---------- paths / device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
assert DATA_PATH.exists(), f"SPR_BENCH not found at {DATA_PATH}"
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helpers ----------
def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.split() if tok})


def count_color_variety(seq: str):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, yt, yp in zip(w, y_true, y_pred) if yt == yp) / max(sum(w), 1)


def harmonic_weighted_accuracy(seqs, y_true, y_pred):
    swa, cwa = shape_weighted_accuracy(seqs, y_true, y_pred), color_weighted_accuracy(
        seqs, y_true, y_pred
    )
    return 0 if (swa + cwa) == 0 else 2 * swa * cwa / (swa + cwa)


class Vocab:
    def __init__(self, tokens: List[str]):
        self.itos = ["<pad>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, toks: List[str]):
        return [self.stoi[t] for t in toks]


class BagClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, num_cls):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_sz, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_cls)

    def forward(self, txt, offs):
        return self.fc(self.embedding(txt, offs))


# ---------- load data / vocab ----------
spr = load_spr_bench(DATA_PATH)
all_tokens = [tok for seq in spr["train"]["sequence"] for tok in seq.split()]
vocab = Vocab(all_tokens)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}


def collate(batch):
    token_ids, offsets, label_ids = [], [0], []
    for ex in batch:
        ids = vocab(ex["sequence"].split())
        token_ids.extend(ids)
        offsets.append(offsets[-1] + len(ids))
        label_ids.append(label2id[ex["label"]])
    return (
        torch.tensor(token_ids, dtype=torch.long).to(device),
        torch.tensor(offsets[:-1], dtype=torch.long).to(device),
        torch.tensor(label_ids, dtype=torch.long).to(device),
    )


batch_size = 128
train_loader = DataLoader(spr["train"], batch_size, True, collate_fn=collate)
dev_loader = DataLoader(spr["dev"], batch_size, False, collate_fn=collate)
test_loader = DataLoader(spr["test"], batch_size, False, collate_fn=collate)

# ---------- evaluation ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    y_t, y_p, seqs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for b_idx, (txt, offs, lbs) in enumerate(loader):
            out = model(txt, offs)
            loss_sum += criterion(out, lbs).item() * lbs.size(0)
            preds = out.argmax(1).cpu().tolist()
            y_p.extend([id2label[p] for p in preds])
            y_t.extend([id2label[i] for i in lbs.cpu().tolist()])
            s_start = b_idx * batch_size
            s_end = s_start + lbs.size(0)
            seqs.extend(loader.dataset["sequence"][s_start:s_end])
    n = len(y_t)
    return (
        loss_sum / n,
        shape_weighted_accuracy(seqs, y_t, y_p),
        color_weighted_accuracy(seqs, y_t, y_p),
        harmonic_weighted_accuracy(seqs, y_t, y_p),
        y_t,
        y_p,
    )


# ---------- experiment dict ----------
experiment_data = {"max_grad_norm": {"SPR_BENCH": {}}}

# ---------- hyperparameter sweep ----------
clip_values = [0, 0.5, 1, 2, 5]
epochs = 5
embed_dim = 64
lr = 1e-3

for clip in clip_values:
    set_seed(42)  # reproducibility per run
    model = BagClassifier(len(vocab), embed_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    run_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # training
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for txt, offs, lbs in train_loader:
            optimizer.zero_grad()
            out = model(txt, offs)
            loss = criterion(out, lbs)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += loss.item() * lbs.size(0)
        train_loss = running / len(spr["train"])
        val_loss, swa, cwa, hwa, _, _ = evaluate(model, dev_loader)
        run_rec["losses"]["train"].append(train_loss)
        run_rec["losses"]["val"].append(val_loss)
        run_rec["metrics"]["train"].append(None)
        run_rec["metrics"]["val"].append({"SWA": swa, "CWA": cwa, "HWA": hwa})
        print(
            f"clip={clip} | epoch={ep} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | SWA={swa:.4f} | CWA={cwa:.4f} | HWA={hwa:.4f}"
        )
    # final test
    t_loss, swa_t, cwa_t, hwa_t, y_true_t, y_pred_t = evaluate(model, test_loader)
    print(
        f"[clip {clip}] Test | loss={t_loss:.4f} | SWA={swa_t:.4f} | "
        f"CWA={cwa_t:.4f} | HWA={hwa_t:.4f}"
    )
    run_rec["predictions"] = y_pred_t
    run_rec["ground_truth"] = y_true_t
    run_rec["test_metrics"] = {"loss": t_loss, "SWA": swa_t, "CWA": cwa_t, "HWA": hwa_t}
    experiment_data["max_grad_norm"]["SPR_BENCH"][str(clip)] = run_rec

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
