import os, pathlib, math, time, json, random, itertools, numpy as np, torch, gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

# ---------------------- house-keeping -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# master log
experiment_data = {"batch_size_tuning": {}}


# ---------------------- metric helpers ------------------------------
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


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# ---------------------- data loading --------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path(
    os.getenv("SPR_BENCH_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ---------------------- vocab & encoders ----------------------------
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


def build_vocab(dataset):
    toks = set()
    for seq in dataset["sequence"]:
        toks.update(seq.strip().split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok in sorted(toks):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
print("Vocab size:", len(vocab))


def encode_sequence(seq, vocab=vocab):
    return [vocab.get(t, vocab[UNK_TOKEN]) for t in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
label2idx = {l: i for i, l in enumerate(label_set)}
idx2label = {i: l for l, i in label2idx.items()}


class SPRTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2idx[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            "seq_enc": torch.tensor(encode_sequence(self.seqs[i]), dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "raw_seq": self.seqs[i],
        }


def collate_fn(batch):
    enc = [b["seq_enc"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    padded = nn.utils.rnn.pad_sequence(
        enc, batch_first=True, padding_value=vocab[PAD_TOKEN]
    )
    return {"input_ids": padded, "labels": labels, "raw_seq": raw}


train_ds = SPRTorchDataset(spr["train"])
dev_ds = SPRTorchDataset(spr["dev"])
test_ds = SPRTorchDataset(spr["test"])

# Precompute train rule signatures for NRGS
train_signatures = set(rule_signature(s) for s in spr["train"]["sequence"])


# ---------------------- model ---------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


# ---------------------- evaluation ----------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_seq, all_true, all_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inp)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * len(labels)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            tot += len(labels)
            all_seq.extend(batch["raw_seq"])
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    acc = correct / tot
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    cwa = color_weighted_accuracy(all_seq, all_true, all_pred)
    novel_mask = [rule_signature(s) not in train_signatures for s in all_seq]
    novel_total = sum(novel_mask)
    novel_correct = sum(
        int(p == t) for p, t, m in zip(all_pred, all_true, novel_mask) if m
    )
    nrgs = novel_correct / novel_total if novel_total > 0 else 0.0
    return loss_sum / tot, acc, swa, cwa, nrgs, all_pred, all_true, all_seq


# ---------------------- hyperparameter sweep ------------------------
BATCH_SIZES = [32, 64, 128, 256]
EPOCHS = 5

for bs in BATCH_SIZES:
    print("\n========== Batch size", bs, "=========")
    subdir = os.path.join(working_dir, f"bs_{bs}")
    os.makedirs(subdir, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=min(4 * bs, 512), shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=min(4 * bs, 512), shuffle=False, collate_fn=collate_fn
    )

    model = GRUClassifier(len(vocab), emb=32, hidden=64, num_labels=len(label_set)).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["input_ids"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch["labels"].size(0)
        train_loss = epoch_loss / len(train_ds)

        val_loss, val_acc, val_swa, val_cwa, val_nrgs, _, _, _ = evaluate(
            model, dev_loader, criterion
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} SWA={val_swa:.3f} CWA={val_cwa:.3f} NRGS={val_nrgs:.3f}"
        )

        run_log["losses"]["train"].append(train_loss)
        run_log["losses"]["val"].append(val_loss)
        run_log["metrics"]["train"].append({"epoch": epoch})
        run_log["metrics"]["val"].append(
            {
                "epoch": epoch,
                "acc": val_acc,
                "swa": val_swa,
                "cwa": val_cwa,
                "nrgs": val_nrgs,
            }
        )
        run_log["timestamps"].append(time.time())

    # final test
    test_loss, test_acc, test_swa, test_cwa, test_nrgs, preds, trues, seqs = evaluate(
        model, test_loader, criterion
    )
    print(
        "TEST  loss={:.4f} acc={:.3f} SWA={:.3f} CWA={:.3f} NRGS={:.3f}".format(
            test_loss, test_acc, test_swa, test_cwa, test_nrgs
        )
    )

    run_log["predictions"] = preds
    run_log["ground_truth"] = trues
    run_log["metrics"]["test"] = {
        "loss": test_loss,
        "acc": test_acc,
        "swa": test_swa,
        "cwa": test_cwa,
        "nrgs": test_nrgs,
    }

    # save sub-plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Acc", "SWA", "CWA", "NRGS"],
        [test_acc, test_swa, test_cwa, test_nrgs],
        color="skyblue",
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"SPR_BENCH Test Metrics (bs={bs})")
    plt.tight_layout()
    plot_path = os.path.join(subdir, "spr_metrics_bar.png")
    plt.savefig(plot_path)
    plt.close()
    print("Plot saved to", plot_path)

    # persist run log and free gpu mem
    experiment_data["batch_size_tuning"][f"bs_{bs}"] = run_log
    del model, optimizer, train_loader, dev_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------- save aggregated artefacts -------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
