# -------------- imports --------------
import os, pathlib, random, time, numpy as np, torch, math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# -------------- working dir --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- device --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------- metrics --------------
def count_shape(sequence):
    return len(set(tok[0] for tok in sequence.split() if tok))


def count_color(sequence):
    return len(set(tok[1] for tok in sequence.split() if len(tok) > 1))


def CRWA(seqs, y_true, y_pred):
    w = [count_shape(s) * count_color(s) for s in seqs]
    corr = [a if t == p else 0 for a, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def SWA(seqs, y_true, y_pred):
    w = [count_shape(s) for s in seqs]
    corr = [a if t == p else 0 for a, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def CWA(seqs, y_true, y_pred):
    w = [count_color(s) for s in seqs]
    corr = [a if t == p else 0 for a, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


# -------------- dataset loading --------------
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
        dset = load_spr_bench(
            pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
        )
        print("Loaded real SPR_BENCH dataset.")
    except Exception as e:
        print("Generating synthetic toy data due to:", e)

        def gen(n):
            shapes = "ABCD"
            colors = "abcd"
            data = []
            for i in range(n):
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(3, 10))
                )
                label = random.choice(["yes", "no"])
                data.append({"id": i, "sequence": seq, "label": label})
            return load_dataset("json", data_files={"train": data}, split="train")

        dset = DatasetDict()
        dset["train"], dset["dev"], dset["test"] = gen(1000), gen(200), gen(200)
    return dset


spr = get_dataset()

# -------------- vocab + labels --------------
all_tokens = set()
all_labels = set()
for ex in spr["train"]:
    all_tokens.update(ex["sequence"].split())
    all_labels.add(ex["label"])
tok2id = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}  # 0 PAD
label2id = {lab: i for i, lab in enumerate(sorted(all_labels))}
id2label = {v: k for k, v in label2id.items()}
vocab_size = len(tok2id) + 1
num_classes = len(label2id)
print(f"Vocab size={vocab_size-1}, classes={num_classes}")


# -------------- torch dataset --------------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seq = hf_split["sequence"]
        self.lbl = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.seq[idx]
        toks = s.split()
        ids = [tok2id[t] for t in toks]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(len(ids), dtype=torch.long),
            "n_shape": torch.tensor(count_shape(s), dtype=torch.long),
            "n_color": torch.tensor(count_color(s), dtype=torch.long),
            "label": torch.tensor(self.lbl[idx], dtype=torch.long),
            "raw_seq": s,
        }


def collate(batch):
    max_len = max(b["length"] for b in batch).item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
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


# -------------- model --------------
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


# -------------- evaluation --------------
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, n = 0, 0
    seqs = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([bt["n_shape"], bt["n_color"]], dim=-1).float()
            logits = model(bt["input_ids"], bt["mask"], feat)
            loss = criterion(logits, bt["label"])
            bs = bt["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            preds = logits.argmax(-1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(bt["label"].cpu().numpy())
            seqs.extend(batch["raw_seq"])
    return (
        tot_loss / n,
        CRWA(seqs, y_true, y_pred),
        SWA(seqs, y_true, y_pred),
        CWA(seqs, y_true, y_pred),
        y_true,
        y_pred,
    )


# -------------- experiment data skeleton --------------
experiment_data = {
    "batch_size_tuning": {
        "SPR_BENCH": {
            "batch_sizes": [],
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": [], "test": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
        }
    }
}

# -------------- hyperparameter sweep --------------
embed_dim = 64
epochs = 5
batch_sizes_to_try = [32, 64, 128, 256]
for bs in batch_sizes_to_try:
    print(f"\n=== Training with batch size {bs} ===")
    # dataloaders
    train_loader = DataLoader(
        SPRTorch(spr["train"]), batch_size=bs, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        SPRTorch(spr["dev"]), batch_size=bs, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        SPRTorch(spr["test"]), batch_size=bs, shuffle=False, collate_fn=collate
    )
    # model, optimizer
    model = AvgEmbedClassifier(vocab_size, embed_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses_epoch = []
    val_losses_epoch = []
    val_metrics_epoch = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, n = 0, 0
        for batch in train_loader:
            bt = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            feat = torch.stack([bt["n_shape"], bt["n_color"]], dim=-1).float()
            optimizer.zero_grad()
            logits = model(bt["input_ids"], bt["mask"], feat)
            loss = criterion(logits, bt["label"])
            loss.backward()
            optimizer.step()
            bs_cur = bt["label"].size(0)
            ep_loss += loss.item() * bs_cur
            n += bs_cur
        train_loss = ep_loss / n
        val_loss, crwa, swa, cwa, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"Epoch {ep}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CRWA={crwa:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
        )
        train_losses_epoch.append(train_loss)
        val_losses_epoch.append(val_loss)
        val_metrics_epoch.append({"CRWA": crwa, "SWA": swa, "CWA": cwa})
    # final test
    test_loss, crwa_t, swa_t, cwa_t, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )
    print(
        f"TEST (bs={bs}): loss={test_loss:.4f} | CRWA={crwa_t:.4f} | SWA={swa_t:.4f} | CWA={cwa_t:.4f}"
    )
    # store experiment data
    ed = experiment_data["batch_size_tuning"]["SPR_BENCH"]
    ed["batch_sizes"].append(bs)
    ed["losses"]["train"].append(train_losses_epoch)
    ed["losses"]["val"].append(val_losses_epoch)
    ed["losses"]["test"].append(test_loss)
    ed["metrics"]["val"].append(val_metrics_epoch)
    ed["metrics"]["test"].append({"CRWA": crwa_t, "SWA": swa_t, "CWA": cwa_t})
    ed["predictions"].append(y_pred)
    ed["ground_truth"].append(y_true)
    ed["timestamps"].append(time.time())

# -------------- save --------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nHyperparameter tuning complete. Results saved to 'experiment_data.npy'")
