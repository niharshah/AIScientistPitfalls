import os, pathlib, random, string, numpy as np, torch, math, time
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────── housekeeping / output dict
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
experiment_data = {"batch_size_sweep": {"SPR_BENCH": {}}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ─────────────────────────── helpers from baseline
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    correct = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1)


# ─────────────────────────── dataset loading (real or synthetic)
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real data, generating synthetic.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:5])
        colors = list("12345")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seq = " ".join(toks)
            maj_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            labels.append(maj_shape)
            seqs.append(seq)
        ids = list(range(n))
        return {"id": ids, "sequence": seqs, "label": labels}

    import datasets

    spr = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(synth_split(2000)),
            "dev": datasets.Dataset.from_dict(synth_split(400)),
            "test": datasets.Dataset.from_dict(synth_split(400)),
        }
    )

# ─────────────────────────── vocab build
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


# ─────────────────────────── torch dataset & collate
class SPRTorch(Dataset):
    def __init__(self, hf_dataset):
        self.seqs, self.labels = hf_dataset["sequence"], hf_dataset["label"]

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
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    raws = []
    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        input_ids[i, :l] = item["input_ids"]
        labels[i] = item["label_id"]
        raws.append(item["raw_seq"])
    return {"input_ids": input_ids, "labels": labels, "raw_seq": raws}


# ─────────────────────────── model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, num_labels, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.cls = nn.Linear(dim, num_labels)

    def forward(self, ids):
        emb = self.embed(ids)
        mask = (ids != PAD_ID).unsqueeze(-1)
        summed = (emb * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1)
        return self.cls(summed / lengths)


# ─────────────────────────── sweep over batch sizes
BATCH_SIZES = [32, 64, 128, 256]
EPOCHS = 5

for bs in BATCH_SIZES:
    print(f"\n=== Training with batch_size={bs} ===")
    # loaders
    train_loader = DataLoader(
        SPRTorch(spr["train"]), batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
    )

    model = MeanPoolClassifier(len(token2id) + 1, len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # storage for this config
    config_store = {
        "metrics": {"train_loss": [], "val_loss": [], "SWA": [], "CWA": [], "HWA": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(ids)
        train_loss = running_loss / len(train_loader.dataset)
        # eval
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        seqs = []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(ids)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                y_pred.extend([id2label[p] for p in preds])
                y_true.extend([id2label[l] for l in labs])
                seqs.extend(batch["raw_seq"])
        val_loss /= len(dev_loader.dataset)

        # metrics
        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = 0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)
        config_store["metrics"]["train_loss"].append(train_loss)
        config_store["metrics"]["val_loss"].append(val_loss)
        config_store["metrics"]["SWA"].append(swa)
        config_store["metrics"]["CWA"].append(cwa)
        config_store["metrics"]["HWA"].append(hwa)
        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    # final predictions storage
    config_store["predictions"] = y_pred
    config_store["ground_truth"] = y_true
    experiment_data["batch_size_sweep"]["SPR_BENCH"][str(bs)] = config_store

# ─────────────────────────── save
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("\nSaved experiment_data.npy to", working_dir)
