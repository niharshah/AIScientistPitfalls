import os, pathlib, random, time, math, json, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset

# ----------------------- EXPERIMENT REGISTRY ------------------------- #
experiment_data = {"batch_size": {}}  # top-level key required by spec
SAVE_PATH = "experiment_data.npy"

# --------------------------- ENV / SEED ------------------------------ #
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------- DATA UTILS ------------------------------ #
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def build_vocab(dataset: Dataset, seq_field: str = "sequence"):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for s in dataset[seq_field]:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


def encode_sequence(seq, vocab, max_len=None):
    tokens = [vocab.get(tok, vocab["<unk>"]) for tok in seq.strip().split()]
    if max_len is not None:
        tokens = tokens[:max_len]
    return tokens


# ------------------------ SYNTHETIC DATA ----------------------------- #
def build_synthetic(num_train=500, num_dev=100, num_test=200, seqlen=10, vocab_sz=12):
    symbols = [chr(ord("A") + i) for i in range(vocab_sz)]

    def gen_split(n):
        data = {"id": [], "sequence": [], "label": []}
        for i in range(n):
            seq = [random.choice(symbols) for _ in range(seqlen)]
            label = 1 if seq.count("A") % 2 == 0 else 0
            data["id"].append(str(i))
            data["sequence"].append(" ".join(seq))
            data["label"].append(label)
        return Dataset.from_dict(data)

    return DatasetDict(
        train=gen_split(num_train), dev=gen_split(num_dev), test=gen_split(num_test)
    )


# --------------------------- MODEL ----------------------------------- #
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(512, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        mask_flt = (~mask).unsqueeze(-1)
        h_sum = (h * mask_flt).sum(1)
        lengths = mask_flt.sum(1).clamp(min=1)
        pooled = h_sum / lengths
        return self.classifier(pooled)


# -------------------------- COLLATE FN ------------------------------- #
def collate_fn(batch, vocab, max_len=128):
    seqs = [encode_sequence(b["sequence"], vocab, max_len) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len_b = max(len(s) for s in seqs)
    pad_val = vocab["<pad>"]
    padded = [s + [pad_val] * (max_len_b - len(s)) for s in seqs]
    x = torch.tensor(padded, dtype=torch.long)
    mask = x == pad_val
    return {"input_ids": x, "attention_mask": mask, "labels": labels}


# ------------------------ DATA LOADING ------------------------------- #
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
try:
    datasets_dict = load_spr_bench(DATA_PATH)
    print("Loaded real SPR_BENCH dataset.")
except Exception as e:
    print("Could not load real dataset, generating synthetic:", e)
    datasets_dict = build_synthetic()

dataset_name = "SPR_BENCH"
experiment_data["batch_size"][dataset_name] = {}

vocab = build_vocab(datasets_dict["train"])
num_classes = len(set(datasets_dict["train"]["label"]))
print(f"Vocab size: {len(vocab)}, Num classes: {num_classes}")

# -------------------- TRAINING / TUNING LOOP ------------------------- #
batch_sizes = [16, 32, 64, 128, 256]
epochs = 5
for bs in batch_sizes:
    print(f"\n--- Training with batch_size={bs} ---")
    # dataloaders
    train_dl = DataLoader(
        datasets_dict["train"],
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab),
    )
    dev_dl = DataLoader(
        datasets_dict["dev"],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab),
    )
    test_dl = DataLoader(
        datasets_dict["test"],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab),
    )

    # model / optimiser
    model = SimpleTransformerClassifier(
        len(vocab), 128, 4, 2, num_classes, vocab["<pad>"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # experiment record slot
    exp_rec = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    experiment_data["batch_size"][dataset_name][bs] = exp_rec

    # training epochs
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, correct, count = 0.0, 0, 0
        for batch in train_dl:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            count += batch["labels"].size(0)
        train_loss = tot_loss / count
        train_acc = correct / count

        # validation
        model.eval()
        with torch.no_grad():
            tot_loss_v, corr_v, cnt_v = 0.0, 0, 0
            for batch in dev_dl:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])
                tot_loss_v += loss.item() * batch["labels"].size(0)
                preds = logits.argmax(-1)
                corr_v += (preds == batch["labels"]).sum().item()
                cnt_v += batch["labels"].size(0)
        val_loss = tot_loss_v / cnt_v
        val_acc = corr_v / cnt_v

        exp_rec["metrics"]["train"].append(train_acc)
        exp_rec["metrics"]["val"].append(val_acc)
        exp_rec["losses"]["train"].append(train_loss)
        exp_rec["losses"]["val"].append(val_loss)

        print(
            f"Epoch {ep}/{epochs} | "
            f"Train Acc {train_acc:.3f} Loss {train_loss:.3f} | "
            f"Val Acc {val_acc:.3f} Loss {val_loss:.3f}"
        )

    # ----------- Final test evaluation for this batch size ------------ #
    model.eval()
    preds_all, gts_all, tot_loss_t, corr_t, cnt_t = [], [], 0.0, 0, 0
    with torch.no_grad():
        for batch in test_dl:
            gts_all.extend(batch["labels"].tolist())
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch_gpu["input_ids"], batch_gpu["attention_mask"])
            preds = logits.argmax(-1).cpu().tolist()
            preds_all.extend(preds)
            loss = criterion(logits, batch_gpu["labels"])
            tot_loss_t += loss.item() * batch_gpu["labels"].size(0)
            corr_t += (logits.argmax(-1) == batch_gpu["labels"]).sum().item()
            cnt_t += batch_gpu["labels"].size(0)
    test_acc = corr_t / cnt_t
    print(f"Test Acc for batch_size {bs}: {test_acc:.3f}")

    exp_rec["predictions"] = preds_all
    exp_rec["ground_truth"] = gts_all

# --------------------------- SAVE RESULT ----------------------------- #
os.makedirs("working", exist_ok=True)
np.save(os.path.join("working", SAVE_PATH), experiment_data)
print(f"\nSaved all experiment data to working/{SAVE_PATH}")
