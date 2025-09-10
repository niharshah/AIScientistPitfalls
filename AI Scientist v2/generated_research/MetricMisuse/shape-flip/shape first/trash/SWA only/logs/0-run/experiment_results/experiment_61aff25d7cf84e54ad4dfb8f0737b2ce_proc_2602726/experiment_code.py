import os, random, string, datetime, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------- house-keeping ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"batch_size_tuning": {}}  # top-level container
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------- synthetic / real data -------------------
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


if not spr_files_exist(SPR_PATH):
    print("Real SPR_BENCH not found – creating synthetic.")
    shapes = list(string.ascii_uppercase[:6])
    colors = list(map(str, range(4)))

    def random_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(seq):
        u_shapes = len(set(tok[0] for tok in seq.split()))
        u_colors = len(set(tok[1] for tok in seq.split()))
        return int(u_shapes == u_colors)

    def make_split(n):
        s = [random_seq() for _ in range(n)]
        return {"sequence": s, "label": [rule(x) for x in s]}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    print("Loading real SPR_BENCH")
    from datasets import load_dataset, DatasetDict

    def load_csv(split):
        return load_dataset(
            "csv",
            data_files=os.path.join(SPR_PATH, f"{split}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict({sp: load_csv(sp) for sp in ["train", "dev", "test"]})
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }

# ------------------------------ helpers -------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    vocab.update(
        {
            tok: i + 2
            for i, tok in enumerate(sorted({t for s in seqs for t in s.split()}))
        }
    )
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vsize = len(vocab)


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.split()]


class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw = seqs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"ids": self.X[i], "label": self.y[i]}


def collate(batch):
    lens = [len(b["ids"]) for b in batch]
    m = max(lens)
    ids = torch.full((len(batch), m), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : lens[i]] = b["ids"]
    return {
        "ids": ids,
        "lens": torch.tensor(lens),
        "labels": torch.tensor([b["label"] for b in batch]),
    }


def swa(seqs, ytrue, ypred):
    w = [len(set(tok[0] for tok in s.split())) for s in seqs]
    return sum(wi for wi, t, p in zip(w, ytrue, ypred) if t == p) / (sum(w) or 1)


def cwa(seqs, ytrue, ypred):
    w = [len(set(tok[1] for tok in s.split())) for s in seqs]
    return sum(wi for wi, t, p in zip(w, ytrue, ypred) if t == p) / (sum(w) or 1)


def signatures(seqs):
    sig = []
    for s in seqs:
        sig.append(
            (
                tuple(sorted(set(tok[0] for tok in s.split()))),
                tuple(sorted(set(tok[1] for tok in s.split()))),
            )
        )
    return sig


class GRUClassifier(nn.Module):
    def __init__(self, vs, ed=64, hd=128, nc=2):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=vocab[PAD])
        self.gru = nn.GRU(ed, hd, batch_first=True)
        self.fc = nn.Linear(hd, nc)

    def forward(self, x, l):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h.squeeze(0))


datasets = {
    sp: SPRDataset(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}
num_classes = len(set(raw_data["train"]["label"]))


def train_for_batch_size(bs, epochs=6, lr=1e-3):
    print(f"\n=== Training with batch_size={bs} ===")
    model = GRUClassifier(vsize, nc=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    loaders = {
        sp: DataLoader(
            datasets[sp], batch_size=bs, shuffle=(sp == "train"), collate_fn=collate
        )
        for sp in ["train", "dev", "test"]
    }

    logs = {
        "losses": {"train": [], "dev": []},
        "metrics": {"train": [], "dev": [], "test": [], "NRGS": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    def evaluate(split):
        model.eval()
        tot = 0
        correct = 0
        loss_sum = 0
        preds = []
        with torch.no_grad():
            for b in loaders[split]:
                b = {k: v.to(device) for k, v in b.items()}
                out = model(b["ids"], b["lens"])
                loss = crit(out, b["labels"])
                p = out.argmax(-1)
                preds.extend(p.cpu().tolist())
                correct += (p == b["labels"]).sum().item()
                tot += b["labels"].size(0)
                loss_sum += loss.item() * b["labels"].size(0)
        acc = correct / tot
        return acc, loss_sum / tot, preds

    # training loop
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0
        for b in loaders["train"]:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["ids"], b["lens"])
            loss = crit(out, b["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * b["labels"].size(0)
        avg_train_loss = run_loss / len(datasets["train"])
        train_acc, _, _ = evaluate("train")
        dev_acc, dev_loss, _ = evaluate("dev")
        logs["losses"]["train"].append(avg_train_loss)
        logs["losses"]["dev"].append(dev_loss)
        logs["metrics"]["train"].append({"acc": train_acc})
        logs["metrics"]["dev"].append({"acc": dev_acc})
        logs["timestamps"].append(str(datetime.datetime.now()))
        print(
            f"Epoch {ep}: train_loss={avg_train_loss:.4f} val_loss={dev_loss:.4f} val_acc={dev_acc:.3f}"
        )

    # final test eval & NRGS
    test_acc, _, test_preds = evaluate("test")
    logs["metrics"]["test"] = {"acc": test_acc}
    logs["predictions"] = test_preds
    logs["ground_truth"] = datasets["test"].y.tolist()

    train_sigs = set(signatures(raw_data["train"]["sequence"]))
    test_sigs = signatures(raw_data["test"]["sequence"])
    novel = [i for i, s in enumerate(test_sigs) if s not in train_sigs]
    NRGS = sum(1 for i in novel if test_preds[i] == logs["ground_truth"][i]) / (
        len(novel) or 1
    )
    logs["metrics"]["NRGS"] = NRGS
    print(f"TEST acc={test_acc:.3f}  NRGS={NRGS:.3f}")

    # plot losses
    plt.figure()
    plt.plot(logs["losses"]["train"], label="train")
    plt.plot(logs["losses"]["dev"], label="dev")
    plt.title(f"Loss (bs={bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_bs{bs}.png"))
    plt.close()
    return logs


batch_sizes = [32, 64, 128]
for bs in batch_sizes:
    logs = train_for_batch_size(bs)
    experiment_data["batch_size_tuning"][f"bs_{bs}"] = logs

# ----------------------- persist experiment data ----------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)
print("\nAll experiments completed. Data saved to working/experiment_data.npy")
