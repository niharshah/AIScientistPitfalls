import os, random, string, datetime, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------- house-keeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {
    "bidirectional": {"SPR_BENCH": {}}  # hyperparam tuning type  # dataset name
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device :", device)

# ----------------------------- data loading ------------------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{split}.csv"))
        for split in ["train", "dev", "test"]
    )


use_synthetic = not spr_files_exist(SPR_PATH)

if use_synthetic:
    print("Real SPR_BENCH not found – generating synthetic data.")
    shapes = list(string.ascii_uppercase[:6])  # A-F
    colors = [str(i) for i in range(4)]  # 0-3

    def random_seq():
        length = random.randint(4, 9)
        return " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )

    def rule_label(seq):
        us = len(set(tok[0] for tok in seq.split()))
        uc = len(set(tok[1] for tok in seq.split()))
        return int(us == uc)

    def make_split(n):
        seqs = [random_seq() for _ in range(n)]
        labels = [rule_label(s) for s in seqs]
        return {"sequence": seqs, "label": labels}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    from datasets import load_dataset, DatasetDict

    def load_spr_bench(root):
        def _load(split_csv):
            return load_dataset(
                "csv",
                data_files=os.path.join(root, split_csv),
                split="train",
                cache_dir=".cache_dsets",
            )

        d = DatasetDict()
        for sp in ["train", "dev", "test"]:
            d[sp] = _load(f"{sp}.csv")
        return d

    ds = load_spr_bench(SPR_PATH)
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }

# ----------------------------- helpers -----------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    toks = {tok for s in seqs for tok in s.split()}
    vocab.update({t: i + 2 for i, t in enumerate(sorted(toks))})
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vocab_size = len(vocab)
print("Vocab size :", vocab_size)


def encode_sequence(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRTorchDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = [torch.tensor(encode_sequence(s), dtype=torch.long) for s in sequences]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw_seq = sequences

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_ids": self.X[idx], "label": self.y[idx]}


def collate(batch):
    lengths = [len(item["input_ids"]) for item in batch]
    maxlen = max(lengths)
    input_ids = torch.full(
        (len(batch), maxlen), fill_value=vocab[PAD], dtype=torch.long
    )
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = ids
        labels[i] = item["label"]
    return {"input_ids": input_ids, "labels": labels, "lengths": torch.tensor(lengths)}


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) or 1
    )


def compute_signatures(seqs):
    sigs = []
    for s in seqs:
        shapes = tuple(sorted(set(tok[0] for tok in s.split())))
        colors = tuple(sorted(set(tok[1] for tok in s.split())))
        sigs.append((shapes, colors))
    return sigs


# ----------------------------- model -------------------------------------
class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_classes, bidirectional=False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.out = nn.Linear(out_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (num_directions, B, H)
        if self.bidirectional:
            cat = torch.cat((h[0], h[1]), dim=1)  # (B, 2H)
            logits = self.out(cat)
        else:
            logits = self.out(h.squeeze(0))
        return logits


# ------------------------- training / evaluation -------------------------
datasets = {
    sp: SPRTorchDataset(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}
batch_size = 64
loaders_template = lambda: {
    sp: DataLoader(
        datasets[sp], batch_size=batch_size, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ["train", "dev", "test"]
}


def evaluate(model, loaders, split):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loaders[split]:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            loss_sum += loss.item() * batch["labels"].size(0)
    acc = correct / total
    # Full-sequence predictions for SWA/CWA
    model.eval()
    pred_list = []
    raw_seq = loaders[split].dataset.raw_seq
    with torch.no_grad():
        for i in range(0, len(raw_seq), batch_size):
            seqs = raw_seq[i : i + batch_size]
            enc = [encode_sequence(s) for s in seqs]
            lengths = torch.tensor([len(x) for x in enc])
            maxlen = lengths.max()
            inp = torch.full((len(enc), maxlen), vocab[PAD], dtype=torch.long)
            for j, row in enumerate(enc):
                inp[j, : len(row)] = torch.tensor(row)
            logits = model(inp.to(device), lengths.to(device))
            pred_list.extend(logits.argmax(-1).cpu().tolist())
    y_true = loaders[split].dataset.y.tolist()
    swa = shape_weighted_accuracy(raw_seq, y_true, pred_list)
    cwa = color_weighted_accuracy(raw_seq, y_true, pred_list)
    return {
        "acc": acc,
        "swa": swa,
        "cwa": cwa,
        "loss": loss_sum / total,
        "preds": pred_list,
        "gts": y_true,
        "seqs": raw_seq,
    }


# ----------------------------- sweep -------------------------------------
num_classes = len(set(raw_data["train"]["label"]))
embed_dim, hidden_dim = 64, 128
epochs = 6
for bi_flag in [False, True]:
    loaders = loaders_template()
    model = GRUClassifier(
        vocab_size, embed_dim, hidden_dim, num_classes, bidirectional=bi_flag
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses_train, losses_dev = [], []
    metrics_dev = []
    timestamps = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loaders["train"]:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch["labels"].size(0)
        avg_train_loss = running_loss / len(datasets["train"])
        losses_train.append(avg_train_loss)

        dev_res = evaluate(model, loaders, "dev")
        losses_dev.append(dev_res["loss"])
        metrics_dev.append(
            {"acc": dev_res["acc"], "swa": dev_res["swa"], "cwa": dev_res["cwa"]}
        )
        timestamps.append(str(datetime.datetime.now()))
        print(
            f"[bi={bi_flag}] Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f} | val_acc={dev_res['acc']:.3f}"
        )

    # final test evaluation & NRGS
    test_res = evaluate(model, loaders, "test")
    train_sigs = set(compute_signatures(raw_data["train"]["sequence"]))
    test_sigs = compute_signatures(test_res["seqs"])
    novel_idx = [i for i, sg in enumerate(test_sigs) if sg not in train_sigs]
    NRGS = (
        sum(1 for i in novel_idx if test_res["preds"][i] == test_res["gts"][i])
        / len(novel_idx)
        if novel_idx
        else 0.0
    )
    print(
        f"[bi={bi_flag}] TEST acc={test_res['acc']:.3f} SWA={test_res['swa']:.3f} "
        f"CWA={test_res['cwa']:.3f} NRGS={NRGS:.3f}"
    )

    # populate experiment_data
    key = str(bi_flag)
    experiment_data["bidirectional"]["SPR_BENCH"][key] = {
        "metrics": {
            "dev": metrics_dev,
            "test": {
                "acc": test_res["acc"],
                "swa": test_res["swa"],
                "cwa": test_res["cwa"],
            },
            "NRGS": NRGS,
        },
        "losses": {"train": losses_train, "dev": losses_dev},
        "predictions": test_res["preds"],
        "ground_truth": test_res["gts"],
        "timestamps": timestamps,
    }

    # save loss curve
    plt.figure()
    plt.plot(losses_train, label="train")
    plt.plot(losses_dev, label="dev")
    plt.title(f"Loss curve (bidirectional={bi_flag})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_SPR_bi_{bi_flag}.png"))
    plt.close()

# ----------------------------- save all ----------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
print("All results saved to", working_dir)
