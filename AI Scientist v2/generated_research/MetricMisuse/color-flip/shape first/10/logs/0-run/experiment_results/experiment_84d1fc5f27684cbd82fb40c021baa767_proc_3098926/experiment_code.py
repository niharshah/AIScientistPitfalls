import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────  reproducibility & folders
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ────────────────────────────────────────────────  experiment data container
experiment_data = {
    "weight_decay": {  # hyperparam tuning type
        "SPR_BENCH": {"runs": []}  # dataset  # each run will be appended here
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ────────────────────────────────────────────────  metric helpers
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


# ────────────────────────────────────────────────  load or build data
try:
    from SPR import load_spr_bench

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real dataset, generating synthetic toy data.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:5])
        colors = list("12345")
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            maj_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            labels.append(maj_shape)
            seqs.append(" ".join(toks))
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

# ────────────────────────────────────────────────  vocab & encoding
PAD_ID, token2id, label2id = 0, {}, {}


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
    return [token2id[tok] for tok in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ────────────────────────────────────────────────  torch dataset/dataloader
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
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    raw_seqs = []
    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        input_ids[i, :l] = item["input_ids"]
        labels[i] = item["label_id"]
        raw_seqs.append(item["raw_seq"])
    return {"input_ids": input_ids, "labels": labels, "raw_seq": raw_seqs}


train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate
)


# ────────────────────────────────────────────────  model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab, num_labels, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.classifier = nn.Linear(dim, num_labels)

    def forward(self, ids):
        emb = self.embed(ids)
        mask = (ids != PAD_ID).unsqueeze(-1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


# ────────────────────────────────────────────────  hyperparameter grid search
weight_decays = [0, 1e-5, 1e-4, 1e-3]
EPOCHS = 5
for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = MeanPoolClassifier(len(token2id) + 1, len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    run_record = {
        "weight_decay": wd,
        "losses": {"train": [], "val": []},
        "metrics": {"SWA": [], "CWA": [], "HWA": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * ids.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        run_record["losses"]["train"].append(train_loss)

        # ---- evaluate ----
        model.eval()
        val_loss, y_true, y_pred, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * ids.size(0)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                y_pred.extend([id2label[p] for p in preds])
                y_true.extend([id2label[l] for l in labs])
                seqs.extend(batch["raw_seq"])
        val_loss /= len(dev_loader.dataset)
        run_record["losses"]["val"].append(val_loss)

        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = 0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)
        run_record["metrics"]["SWA"].append(swa)
        run_record["metrics"]["CWA"].append(cwa)
        run_record["metrics"]["HWA"].append(hwa)

        print(
            f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    # store final epoch predictions and ground truth
    run_record["predictions"] = y_pred
    run_record["ground_truth"] = y_true
    experiment_data["weight_decay"]["SPR_BENCH"]["runs"].append(run_record)
    torch.cuda.empty_cache()

# ────────────────────────────────────────────────  save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy to", working_dir)
