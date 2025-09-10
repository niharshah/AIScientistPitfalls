import os, pathlib, random, string, time, math, numpy as np, torch, datasets
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────  bookkeeping & storage dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {
    "gradient_clipping_max_norm": {  # hyper-param tuning type
        "SPR_BENCH": {}  # dataset name
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ─────────────────────────────  SPR helpers
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(sum(w), 1)


# ─────────────────────────────  load dataset (real or synthetic)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real dataset, generating synthetic toy data.", e)

    def synth_split(n):
        shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
        seqs, labels = [], []
        for _ in range(n):
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            labels.append(
                max(
                    set(t[0] for t in toks), key=lambda s: [t[0] for t in toks].count(s)
                )
            )
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(2000)),
            "dev": datasets.Dataset.from_dict(synth_split(400)),
            "test": datasets.Dataset.from_dict(synth_split(400)),
        }
    )

# ─────────────────────────────  vocab & encoding
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(dataset):
    global token2id, label2id
    tokens, labels = set(), set()
    for seq, lab in zip(dataset["sequence"], dataset["label"]):
        tokens.update(seq.split())
        labels.add(lab)
    token2id = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}
    label2id = {lab: i for i, lab in enumerate(sorted(labels))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):
    return [token2id[t] for t in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ─────────────────────────────  torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_sequence(self.seqs[idx])),
            "label_id": torch.tensor(encode_label(self.labels[idx])),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    mlen = max(len(x["input_ids"]) for x in batch)
    ids = torch.zeros(len(batch), mlen, dtype=torch.long)
    labs, raws = torch.zeros(len(batch), dtype=torch.long), []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        ids[i, :l] = b["input_ids"]
        labs[i] = b["label_id"]
        raws.append(b["raw_seq"])
    return {"input_ids": ids, "labels": labs, "raw_seq": raws}


train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ─────────────────────────────  model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, n_labels, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.cls = nn.Linear(dim, n_labels)

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != PAD_ID).unsqueeze(-1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.cls(pooled)


# ─────────────────────────────  training procedure for a given max_norm
def run_experiment(max_norm, epochs=5, lr=1e-3):
    tag = f"clip_{max_norm}"
    store = {
        "metrics": {"train_loss": [], "val_loss": [], "SWA": [], "CWA": [], "HWA": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = MeanPoolClassifier(len(token2id) + 1, len(label2id)).to(device)
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0
        for batch in train_loader:
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            opt.zero_grad()
            loss = crit(model(ids), labels)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            opt.step()
            running += loss.item() * len(ids)
        tr_loss = running / len(train_loader.dataset)
        store["metrics"]["train_loss"].append(tr_loss)

        model.eval()
        v_loss = 0
        y_true, y_pred, seqs = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(ids)
                loss = crit(logits, labels)
                v_loss += loss.item() * len(ids)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                y_pred.extend([id2label[p] for p in preds])
                y_true.extend([id2label[l] for l in labs])
                seqs.extend(batch["raw_seq"])
        v_loss /= len(dev_loader.dataset)
        store["metrics"]["val_loss"].append(v_loss)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = 0 if swa == 0 or cwa == 0 else 2 * swa * cwa / (swa + cwa)
        store["metrics"]["SWA"].append(swa)
        store["metrics"]["CWA"].append(cwa)
        store["metrics"]["HWA"].append(hwa)
        print(
            f"[{tag}] Epoch {ep}: train={tr_loss:.4f} val={v_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    store["predictions"], store["ground_truth"] = y_pred, y_true
    experiment_data["gradient_clipping_max_norm"]["SPR_BENCH"][tag] = store


# ─────────────────────────────  run grid search
for clip_val in [0, 0.5, 1, 2, 5]:
    run_experiment(clip_val)

# ─────────────────────────────  save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
