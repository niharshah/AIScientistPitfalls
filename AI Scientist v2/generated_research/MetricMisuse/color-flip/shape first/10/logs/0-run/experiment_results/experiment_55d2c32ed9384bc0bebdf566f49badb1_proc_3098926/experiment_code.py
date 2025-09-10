import os, pathlib, random, string, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────  bookkeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"learning_rate": {}}  # mandatory key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ────────────────────────────────────────────────  helpers
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


# ────────────────────────────────────────────────  load / create dataset
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Real dataset unavailable, using toy synthetic:", e)

    def synth_split(n):
        shapes, colors = list(string.ascii_uppercase[:5]), list("12345")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seqs.append(" ".join(toks))
            maj = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
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

# ────────────────────────────────────────────────  vocabulary
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(data):
    global token2id, label2id
    tokens, labels = set(), set()
    for s, l in zip(data["sequence"], data["label"]):
        tokens.update(s.split())
        labels.add(l)
    token2id = {tok: i + 1 for i, tok in enumerate(sorted(tokens))}
    label2id = {lab: i for i, lab in enumerate(sorted(labels))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):
    return [token2id[t] for t in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ────────────────────────────────────────────────  torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seqs, self.labels = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_sequence(self.seqs[idx]), dtype=torch.long
            ),
            "label_id": torch.tensor(encode_label(self.labels[idx]), dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    mlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros(len(batch), mlen, dtype=torch.long)
    labs = torch.zeros(len(batch), dtype=torch.long)
    raws = []
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


# ────────────────────────────────────────────────  model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, n_labels, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.fc = nn.Linear(dim, n_labels)

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != PAD_ID).unsqueeze(-1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled)


# ────────────────────────────────────────────────  training / evaluation
def run_experiment(lr, epochs=5):
    torch.manual_seed(42)
    model = MeanPoolClassifier(len(token2id) + 1, len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    log = {"train_loss": [], "val_loss": [], "SWA": [], "CWA": [], "HWA": []}
    final_pred, final_true = [], []
    for ep in range(1, epochs + 1):
        # train
        model.train()
        running = 0
        for batch in train_loader:
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            opt.zero_grad()
            loss = crit(model(ids), labels)
            loss.backward()
            opt.step()
            running += loss.item() * len(ids)
        tr_loss = running / len(train_loader.dataset)
        log["train_loss"].append(tr_loss)
        # val
        model.eval()
        vloss, y_true, y_pred, seqs = 0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(ids)
                loss = crit(logits, labels)
                vloss += loss.item() * len(ids)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                seqs += batch["raw_seq"]
                y_true += [id2label[x] for x in labs]
                y_pred += [id2label[x] for x in preds]
        v_loss = vloss / len(dev_loader.dataset)
        log["val_loss"].append(v_loss)
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = 0 if swa == 0 or cwa == 0 else 2 * swa * cwa / (swa + cwa)
        log["SWA"].append(swa)
        log["CWA"].append(cwa)
        log["HWA"].append(hwa)
        print(
            f"[lr={lr}] Epoch {ep}: train={tr_loss:.4f} val={v_loss:.4f} SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )
        final_pred, final_true = y_pred, y_true
    return log, final_pred, final_true


# ────────────────────────────────────────────────  hyper-parameter sweep
lr_candidates = [1e-3, 5e-4, 2e-3]
for lr in lr_candidates:
    metrics, preds, gtruth = run_experiment(lr)
    if "SPR_BENCH" not in experiment_data["learning_rate"]:
        experiment_data["learning_rate"]["SPR_BENCH"] = {}
    experiment_data["learning_rate"]["SPR_BENCH"][str(lr)] = {
        "metrics": metrics,
        "predictions": preds,
        "ground_truth": gtruth,
    }
    torch.cuda.empty_cache()

# ────────────────────────────────────────────────  save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
