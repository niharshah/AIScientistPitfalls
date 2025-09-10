import os, pathlib, random, string, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────── house-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "embedding_dim": {  # ← hyper-parameter tuning type
        "SPR_BENCH": {
            "config_values": [32, 64, 128, 256],
            "losses": {"train": [], "val": []},
            "metrics": {"SWA": [], "CWA": [], "HWA": []},
            "predictions": [],  # list-of-list (per dim)
            "ground_truth": [],  # list-of-list (per dim)
        }
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────── SPR helpers
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


# ─────────────────────────────────────────────── load dataset
try:
    from SPR import load_spr_bench  # real data

    DATA_PATH = pathlib.Path("./SPR_BENCH")
    spr = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH.")
except Exception as e:
    print("Could not load real dataset, generating synthetic toy data.", e)

    def synth_split(n):
        shapes = list(string.ascii_uppercase[:5])  # five shapes
        colors = list("12345")  # five colors
        seqs, labels = [], []
        for _ in range(n):
            ln = random.randint(4, 10)
            toks = [random.choice(shapes) + random.choice(colors) for _ in range(ln)]
            seq = " ".join(toks)
            majority_shape = max(
                set(t[0] for t in toks), key=lambda x: [t[0] for t in toks].count(x)
            )
            labels.append(majority_shape)
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

# ─────────────────────────────────────────────── vocab & encoding
PAD_ID = 0
token2id, label2id = {}, {}


def build_vocabs(dataset):
    global token2id, label2id
    tokens, labels = set(), set()
    for seq, lab in zip(dataset["sequence"], dataset["label"]):
        tokens.update(seq.split())
        labels.add(lab)
    token2id = {tok: idx + 1 for idx, tok in enumerate(sorted(tokens))}
    label2id = {lab: idx for idx, lab in enumerate(sorted(labels))}


build_vocabs(spr["train"])
id2label = {v: k for k, v in label2id.items()}


def encode_sequence(seq):
    return [token2id[tok] for tok in seq.split()]


def encode_label(lab):
    return label2id[lab]


# ─────────────────────────────────────────────── torch dataset
class SPRTorch(Dataset):
    def __init__(self, hf_ds):
        self.seq, self.lab = hf_ds["sequence"], hf_ds["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode_sequence(self.seq[idx]), dtype=torch.long),
            "label_id": torch.tensor(encode_label(self.lab[idx]), dtype=torch.long),
            "raw_seq": self.seq[idx],
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


# ─────────────────────────────────────────────── model definition
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, ids):
        emb = self.embed(ids)  # B × L × D
        mask = (ids != PAD_ID).unsqueeze(-1)  # B × L × 1
        summed = (emb * mask).sum(1)  # B × D
        length = mask.sum(1).clamp(min=1)  # B × 1
        pooled = summed / length
        return self.fc(pooled)


# ─────────────────────────────────────────────── train & eval loop
EMBED_DIMS = [32, 64, 128, 256]
EPOCHS = 5
for dim in EMBED_DIMS:
    print(f"\n─── Embedding dim {dim} ───")
    model = MeanPoolClassifier(len(token2id) + 1, len(label2id), dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            optim.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(ids)
        tr_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        vl_loss, y_true, y_pred, seqs = 0.0, [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(ids)
                loss = criterion(logits, labels)
                vl_loss += loss.item() * len(ids)
                preds = logits.argmax(1).cpu().tolist()
                labs = labels.cpu().tolist()
                seqs.extend(batch["raw_seq"])
                y_true.extend([id2label[x] for x in labs])
                y_pred.extend([id2label[x] for x in preds])
        vl_loss /= len(dev_loader.dataset)

        swa = shape_weighted_accuracy(seqs, y_true, y_pred)
        cwa = color_weighted_accuracy(seqs, y_true, y_pred)
        hwa = 0.0 if (swa == 0 or cwa == 0) else 2 * swa * cwa / (swa + cwa)

        # log
        experiment_data["embedding_dim"]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["embedding_dim"]["SPR_BENCH"]["losses"]["val"].append(vl_loss)
        experiment_data["embedding_dim"]["SPR_BENCH"]["metrics"]["SWA"].append(swa)
        experiment_data["embedding_dim"]["SPR_BENCH"]["metrics"]["CWA"].append(cwa)
        experiment_data["embedding_dim"]["SPR_BENCH"]["metrics"]["HWA"].append(hwa)

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} "
            f"SWA={swa:.3f} CWA={cwa:.3f} HWA={hwa:.3f}"
        )

    # store final epoch predictions for this dim
    experiment_data["embedding_dim"]["SPR_BENCH"]["predictions"].append(y_pred)
    experiment_data["embedding_dim"]["SPR_BENCH"]["ground_truth"].append(y_true)

# ─────────────────────────────────────────────── save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy to", working_dir)
