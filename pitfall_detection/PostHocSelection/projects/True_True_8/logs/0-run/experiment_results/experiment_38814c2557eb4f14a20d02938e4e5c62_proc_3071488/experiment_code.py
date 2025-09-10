import os, random, string, pathlib, time, math, json, copy, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, load_dataset

# ------------- reproducibility -------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------- saving dict -------------
experiment_data = {
    "lr_sweep": {}  # a sub-dict will be created for every learning-rate tried
}

# ------------- working dir -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------- dataset utils -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


def build_synthetic_dataset(n_train=2000, n_dev=500, n_test=500, max_len=12):
    def _gen_row():
        length = random.randint(4, max_len)
        seq, label = [], 0
        for _ in range(length):
            shape = random.choice(string.ascii_uppercase[:5])
            color = random.choice("01234")
            seq.append(shape + color)
            label ^= (ord(shape) + int(color)) & 1
        return {
            "id": str(random.randint(0, 1e9)),
            "sequence": " ".join(seq),
            "label": int(label),
        }

    def _many(n):
        return [_gen_row() for _ in range(n)]

    from datasets import Dataset

    return DatasetDict(
        {
            "train": Dataset.from_list(_many(n_train)),
            "dev": Dataset.from_list(_many(n_dev)),
            "test": Dataset.from_list(_many(n_test)),
        }
    )


# ------------- obtain data -------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
    print("Loaded SPR_BENCH from disk.")
else:
    spr = build_synthetic_dataset()
    print("SPR_BENCH folder not found, using synthetic data.")

# ------------- vocab -------------
PAD, UNK = "<pad>", "<unk>"
vocab = {PAD: 0, UNK: 1}
for split in ["train", "dev", "test"]:
    for seq in spr[split]["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
pad_idx = vocab[PAD]

MAX_LEN = 40


def encode_sequence(seq, max_len=MAX_LEN):
    toks = seq.strip().split()[:max_len]
    ids = [vocab.get(t, vocab[UNK]) for t in toks]
    ids += [pad_idx] * (max_len - len(ids))
    return ids


# ------------- torch dataset -------------
class SPRTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        ids = torch.tensor(encode_sequence(row["sequence"]), dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": ids, "labels": label, "sequence": row["sequence"]}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sequence": [b["sequence"] for b in batch],
    }


train_loader_base = DataLoader(
    SPRTorchDataset(spr["train"]), batch_size=128, shuffle=True, collate_fn=collate_fn
)
dev_loader_base = DataLoader(
    SPRTorchDataset(spr["dev"]), batch_size=256, shuffle=False, collate_fn=collate_fn
)


# ------------- model -------------
class MeanEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)  # B,L,D
        mask = (x != pad_idx).unsqueeze(-1).float()  # B,L,1
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.fc(pooled)


# ------------- augmentation -------------
def shape_rename(seq):
    toks = seq.strip().split()
    shapes = list({t[0] for t in toks})
    mapping = {s: random.choice(string.ascii_uppercase) for s in shapes}
    new_toks = [mapping[t[0]] + t[1:] if len(t) > 1 else mapping[t[0]] for t in toks]
    return " ".join(new_toks)


def compute_ACS(model, dataset, max_samples=1000, n_aug=5):
    model.eval()
    consist_sum = 0.0
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            row = dataset[i]
            base_seq, label = row["sequence"], row["label"]
            seq_variants = [base_seq] + [shape_rename(base_seq) for _ in range(n_aug)]
            correct = 0
            for sv in seq_variants:
                ids = (
                    torch.tensor(encode_sequence(sv), dtype=torch.long)
                    .unsqueeze(0)
                    .to(device)
                )
                pred = model(ids).argmax(-1).item()
                if pred == label:
                    correct += 1
            consist_sum += correct / len(seq_variants)
    return consist_sum / min(len(dataset), max_samples)


# ------------- hyperparam sweep -------------
learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
EPOCHS = 5
num_labels = len(set(spr["train"]["label"]))

for lr in learning_rates:
    print(f"\n==== Training with learning_rate = {lr} ====")
    # copy data loaders to reset iterator each run
    train_loader = train_loader_base
    dev_loader = dev_loader_base

    model = MeanEmbedClassifier(len(vocab), 128, num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_key = f"lr_{lr:.0e}" if lr < 1e-3 else f"lr_{lr}"
    experiment_data["lr_sweep"][lr_key] = {
        "metrics": {"train_loss": [], "val_loss": [], "val_ACS": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            # on-the-fly augmentation (50%)
            seqs_aug = [
                shape_rename(s) if random.random() < 0.5 else s
                for s in batch["sequence"]
            ]
            batch["input_ids"] = torch.stack(
                [torch.tensor(encode_sequence(s), dtype=torch.long) for s in seqs_aug]
            )
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        train_loss = tot_loss / len(train_loader)
        experiment_data["lr_sweep"][lr_key]["metrics"]["train_loss"].append(
            (epoch, train_loss)
        )

        # ---- validation ----
        model.eval()
        vloss, preds_all, gts_all = 0.0, [], []
        with torch.no_grad():
            for batch in dev_loader:
                bt = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(bt["input_ids"])
                vloss += criterion(logits, bt["labels"]).item()
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                gts_all.extend(batch["labels"].cpu().tolist())
        vloss /= len(dev_loader)
        experiment_data["lr_sweep"][lr_key]["metrics"]["val_loss"].append(
            (epoch, vloss)
        )
        experiment_data["lr_sweep"][lr_key]["predictions"] = preds_all
        experiment_data["lr_sweep"][lr_key]["ground_truth"] = gts_all

        # ---- ACS ----
        val_ACS = compute_ACS(model, spr["dev"])
        experiment_data["lr_sweep"][lr_key]["metrics"]["val_ACS"].append(
            (epoch, val_ACS)
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={vloss:.4f} | val_ACS={val_ACS:.4f}"
        )

# ------------- save -------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
