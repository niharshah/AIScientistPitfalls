import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment data dict ----------
experiment_data = {
    "learning_rate": {"SPR_BENCH": {}}  # lr-specific sub-dicts will be inserted here
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- metrics ----------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def _weighted_acc(weights, y_true, y_pred):
    corr = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(corr) / max(sum(weights), 1)


def CRWA(seqs, y_true, y_pred):
    w = [count_shape(s) * count_color(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


def SWA(seqs, y_true, y_pred):
    w = [count_shape(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


def CWA(seqs, y_true, y_pred):
    w = [count_color(s) for s in seqs]
    return _weighted_acc(w, y_true, y_pred)


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


def get_dataset():
    try:
        DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        dset = load_spr_bench(DATA_PATH)
        print("Loaded real SPR_BENCH dataset.")
    except Exception as e:
        print("Could not load real data, generating synthetic toy data.", e)

        def gen(n):
            shapes, colors, data = "ABCD", "abcd", []
            for i in range(n):
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(3, 10))
                )
                data.append(
                    {"id": i, "sequence": seq, "label": random.choice(["yes", "no"])}
                )
            return load_dataset("json", data_files={"train": data}, split="train")

        dset = DatasetDict()
        dset["train"], dset["dev"], dset["test"] = gen(1000), gen(200), gen(200)
    return dset


spr = get_dataset()

# ---------- vocab + label mapping ----------
all_tokens, all_labels = set(), set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 = PAD
label2id = {lab: i for i, lab in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size, num_classes = len(tok2id) + 1, len(label2id)
print(f"Vocab size={vocab_size-1}, classes={num_classes}")


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lbl = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        toks = self.seq[idx].split()
        ids = [tok2id[t] for t in toks]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(len(ids), dtype=torch.long),
            "n_shape": torch.tensor(count_shape(self.seq[idx]), dtype=torch.long),
            "n_color": torch.tensor(count_color(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "raw_seq": self.seq[idx],
        }


def collate(batch):
    max_len = max(x["length"] for x in batch).item()
    padded, mask = torch.zeros(len(batch), max_len, dtype=torch.long), torch.zeros(
        len(batch), max_len, dtype=torch.bool
    )
    for i, b in enumerate(batch):
        l = b["length"]
        padded[i, :l] = b["input_ids"]
        mask[i, :l] = 1
    return {
        "input_ids": padded,
        "mask": mask,
        "n_shape": torch.stack([b["n_shape"] for b in batch]),
        "n_color": torch.stack([b["n_color"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


batch_size = 128
train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(spr["test"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab, embed_dim, num_cls):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim + 2, num_cls)

    def forward(self, ids, mask, feat):
        emb = self.emb(ids)
        mask = mask.unsqueeze(-1)
        avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(torch.cat([avg, feat], dim=-1))


# ---------- evaluation fn ----------
criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    loss_total = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
            logits = model(batch_t["input_ids"], batch_t["mask"], feat)
            loss = criterion(logits, batch_t["label"])
            loss_total += loss.item() * batch_t["label"].size(0)
            n += batch_t["label"].size(0)
            preds = logits.argmax(-1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(batch_t["label"].cpu().numpy())
            all_seq.extend(batch["raw_seq"])
    return (
        loss_total / n,
        CRWA(all_seq, all_true, all_pred),
        SWA(all_seq, all_true, all_pred),
        CWA(all_seq, all_true, all_pred),
        all_true,
        all_pred,
        all_seq,
    )


# ---------- hyper-parameter sweep ----------
lrs = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
epochs, embed_dim = 5, 64

for lr in lrs:
    print(f"\n===== TRAINING WITH learning_rate={lr} =====")
    lr_key = f"{lr:.0e}"
    exp_slot = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    # init model/optim
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0
        m = 0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([batch_t["n_shape"], batch_t["n_color"]], dim=-1).float()
            optimizer.zero_grad()
            logits = model(batch_t["input_ids"], batch_t["mask"], feat)
            loss = criterion(logits, batch_t["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch_t["label"].size(0)
            m += batch_t["label"].size(0)
        train_loss = ep_loss / m

        val_loss, val_crwa, val_swa, val_cwa, y_true, y_pred, _ = evaluate(
            model, dev_loader
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={val_crwa:.4f} | SWA={val_swa:.4f} | CWA={val_cwa:.4f}"
        )

        exp_slot["losses"]["train"].append(train_loss)
        exp_slot["losses"]["val"].append(val_loss)
        exp_slot["metrics"]["train"].append(None)
        exp_slot["metrics"]["val"].append(
            {"CRWA": val_crwa, "SWA": val_swa, "CWA": val_cwa}
        )
        exp_slot["predictions"].append(y_pred)
        exp_slot["ground_truth"].append(y_true)
        exp_slot["timestamps"].append(time.time())

    # test evaluation
    test_loss, test_crwa, test_swa, test_cwa, y_true, y_pred, _ = evaluate(
        model, test_loader
    )
    print(
        f"TEST: loss={test_loss:.4f} | CRWA={test_crwa:.4f} | SWA={test_swa:.4f} | CWA={test_cwa:.4f}"
    )
    exp_slot["metrics"]["test"] = {"CRWA": test_crwa, "SWA": test_swa, "CWA": test_cwa}
    exp_slot["losses"]["test"] = test_loss

    # store under experiment_data
    experiment_data["learning_rate"]["SPR_BENCH"][lr_key] = exp_slot

    # free GPU mem for next run
    del model, optimizer
    torch.cuda.empty_cache()

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
