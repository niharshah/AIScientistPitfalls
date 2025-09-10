import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────── house-keeping & reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"EPOCHS": {"SPR_BENCH": {}}}  # top-level container


# ───────────────────────── accuracy helpers (unchanged)
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ───────────────────────── dataset loading
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Falling back to synthetic data.", e)

    def synth_split(n):
        shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
        seqs, labels = [], []
        for _ in range(n):
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            maj = max(set(t[0] for t in toks), key=[t[0] for t in toks].count)
            labels.append(maj)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    import datasets

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(2000)),
            "dev": datasets.Dataset.from_dict(synth_split(400)),
            "test": datasets.Dataset.from_dict(synth_split(400)),
        }
    )

# ───────────────────────── vocab & encoders
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(ds):
    global token2id, label2id
    toks, labs = set(), set()
    for s, l in zip(ds["sequence"], ds["label"]):
        toks.update(s.split())
        labs.add(l)
    token2id = {tok: i + 1 for i, tok in enumerate(sorted(toks))}
    label2id = {lab: i for i, lab in enumerate(sorted(labs))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):
    return [token2id[t] for t in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ───────────────────────── torch datasets
class SPRTorch(Dataset):
    def __init__(self, hf):
        self.seqs, self.labs = hf["sequence"], hf["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(encode_sequence(self.seqs[i]), dtype=torch.long),
            "label_id": torch.tensor(encode_label(self.labs[i]), dtype=torch.long),
            "raw_seq": self.seqs[i],
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    inp = torch.zeros(len(batch), max_len, dtype=torch.long)
    lab = torch.zeros(len(batch), dtype=torch.long)
    raws = []
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        inp[i, :l] = b["input_ids"]
        lab[i] = b["label_id"]
        raws.append(b["raw_seq"])
    return {"input_ids": inp, "labels": lab, "raw_seq": raws}


train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ───────────────────────── model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, num_labels, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, x):
        e = self.emb(x)
        mask = (x != PAD_ID).unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled)


# ───────────────────────── training function
def train_for_epochs(num_epochs):
    model = MeanPoolClassifier(len(token2id) + 1, len(label2id)).to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    record = {
        "metrics": {"train_loss": [], "val_loss": [], "SWA": [], "CWA": [], "HWA": []},
        "predictions": [],
        "ground_truth": [],
    }

    for ep in range(1, num_epochs + 1):
        # train
        model.train()
        run_loss = 0
        for batch in train_loader:
            ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
            optim.zero_grad()
            logits = model(ids)
            loss = crit(logits, lbl)
            loss.backward()
            optim.step()
            run_loss += loss.item() * len(ids)
        train_loss = run_loss / len(train_loader.dataset)

        # validate
        model.eval()
        vloss, y_t, y_p, seqs = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids, lbl = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(ids)
                loss = crit(logits, lbl)
                vloss += loss.item() * len(ids)
                preds = logits.argmax(1).cpu().tolist()
                y_t.extend([id2label[i] for i in lbl.cpu().tolist()])
                y_p.extend([id2label[i] for i in preds])
                seqs.extend(batch["raw_seq"])
        val_loss = vloss / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(seqs, y_t, y_p)
        cwa = color_weighted_accuracy(seqs, y_t, y_p)
        hwa = 0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)
        record["metrics"]["train_loss"].append(train_loss)
        record["metrics"]["val_loss"].append(val_loss)
        record["metrics"]["SWA"].append(swa)
        record["metrics"]["CWA"].append(cwa)
        record["metrics"]["HWA"].append(hwa)
        print(
            f"[{num_epochs}-ep run] Epoch {ep:02}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )
    # store last predictions
    record["predictions"], record["ground_truth"] = y_p, y_t
    del model
    torch.cuda.empty_cache()
    return record


# ───────────────────────── hyperparameter sweep
epoch_choices = [5, 10, 15, 20, 30]
for ep_cap in epoch_choices:
    print("\n" + "=" * 10, f"Training for {ep_cap} epochs", "=" * 10)
    result = train_for_epochs(ep_cap)
    experiment_data["EPOCHS"]["SPR_BENCH"][f"{ep_cap}_epochs"] = result

# ───────────────────────── save artefacts
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy to", working_dir)
