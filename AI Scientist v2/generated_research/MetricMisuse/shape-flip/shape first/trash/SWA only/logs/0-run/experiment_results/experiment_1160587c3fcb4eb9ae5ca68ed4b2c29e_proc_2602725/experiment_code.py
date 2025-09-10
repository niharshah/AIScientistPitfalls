# hyperparam-tuning_embed_dim.py
import os, random, string, datetime, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# House-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"embed_dim": {}}  # <- top level key
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ------------------------------------------------------------------ #
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# ------------------------------------------------------------------ #
# Load / create SPR_BENCH (synthetic fallback identical to baseline)
SPR_PATH = os.environ.get("SPR_PATH", "./SPR_BENCH")


def spr_files_exist(path):
    return all(
        os.path.isfile(os.path.join(path, f"{sp}.csv"))
        for sp in ["train", "dev", "test"]
    )


use_synth = not spr_files_exist(SPR_PATH)
if use_synth:
    print("Generating synthetic SPR_BENCH")
    shapes = list(string.ascii_uppercase[:6])
    colors = [str(i) for i in range(4)]

    def rand_seq():
        return " ".join(
            random.choice(shapes) + random.choice(colors)
            for _ in range(random.randint(4, 9))
        )

    def rule(seq):
        us = len(set(t[0] for t in seq.split()))
        uc = len(set(t[1] for t in seq.split()))
        return int(us == uc)

    def make_split(n):
        s = [rand_seq() for _ in range(n)]
        return {"sequence": s, "label": [rule(x) for x in s]}

    raw_data = {
        "train": make_split(2000),
        "dev": make_split(400),
        "test": make_split(600),
    }
else:
    from datasets import load_dataset, DatasetDict

    def load(root, csv):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    ds = DatasetDict(
        {sp: load(SPR_PATH, f"{sp}.csv") for sp in ["train", "dev", "test"]}
    )
    raw_data = {
        sp: {"sequence": ds[sp]["sequence"], "label": ds[sp]["label"]}
        for sp in ["train", "dev", "test"]
    }

# ------------------------------------------------------------------ #
# Helpers / metrics identical to baseline
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(seqs):
    vocab = {PAD: 0, UNK: 1}
    vocab.update(
        {
            t: i + 2
            for i, t in enumerate(sorted({tok for s in seqs for tok in s.split()}))
        }
    )
    return vocab


vocab = build_vocab(raw_data["train"]["sequence"])
vocab_size = len(vocab)


def encode(seq):
    return [vocab.get(t, vocab[UNK]) for t in seq.split()]


def count_shape(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color(seq):
    return len(set(tok[1] for tok in seq.split()))


def swa(seqs, y, g):
    w = [count_shape(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y, g)) / (sum(w) or 1)


def cwa(seqs, y, g):
    w = [count_color(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y, g)) / (sum(w) or 1)


def signatures(seqs):
    out = []
    for s in seqs:
        out.append(
            (
                tuple(sorted({t[0] for t in s.split()})),
                tuple(sorted({t[1] for t in s.split()})),
            )
        )
    return out


# ------------------------------------------------------------------ #
# Dataset / dataloader
class SPRDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = [torch.tensor(encode(s), dtype=torch.long) for s in seqs]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.raw = seqs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.X[i], "label": self.y[i]}


def collate(batch):
    lens = [len(b["input_ids"]) for b in batch]
    m = max(lens)
    ids = torch.full((len(batch), m), vocab[PAD], dtype=torch.long)
    lab = torch.empty(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
        lab[i] = b["label"]
    return {"input_ids": ids, "labels": lab, "lengths": torch.tensor(lens)}


datasets = {
    sp: SPRDataset(raw_data[sp]["sequence"], raw_data[sp]["label"])
    for sp in ["train", "dev", "test"]
}


# ------------------------------------------------------------------ #
# Model definition
class GRUClassifier(nn.Module):
    def __init__(self, V, E, H, C):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=vocab[PAD])
        self.gru = nn.GRU(E, H, batch_first=True)
        self.out = nn.Linear(H, C)

    def forward(self, x, lens):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.out(h.squeeze(0))


# ------------------------------------------------------------------ #
# Training / evaluation routine
def run_experiment(embed_dim, epochs=6, batch_size=64):
    print(f"\n===== embed_dim={embed_dim} =====")
    model = GRUClassifier(
        vocab_size, embed_dim, 128, len(set(raw_data["train"]["label"]))
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loaders = {
        sp: DataLoader(
            datasets[sp],
            batch_size=batch_size,
            shuffle=(sp == "train"),
            collate_fn=collate,
        )
        for sp in ["train", "dev", "test"]
    }
    losses = {"train": [], "dev": []}
    metrics_dev = []

    def evaluate(split):
        model.eval()
        correct = tot = lsum = 0
        preds = []
        with torch.no_grad():
            for batch in loaders[split]:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                logits = model(batch["input_ids"], batch["lengths"])
                loss = crit(logits, batch["labels"])
                pr = logits.argmax(-1)
                correct += (pr == batch["labels"]).sum().item()
                tot += pr.size(0)
                lsum += loss.item() * pr.size(0)
                preds.append(pr.cpu())
        acc = correct / tot
        loss_avg = lsum / tot
        preds = torch.cat(preds).tolist()
        seqs = datasets[split].raw
        gts = datasets[split].y.tolist()
        return acc, loss_avg, preds, gts, seqs

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0
        for batch in loaders["train"]:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["input_ids"], batch["lengths"])
            loss = crit(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * batch["labels"].size(0)
        train_loss = run_loss / len(datasets["train"])
        losses["train"].append(train_loss)

        dev_acc, dev_loss, _, _, _ = evaluate("dev")
        losses["dev"].append(dev_loss)
        metrics_dev.append(dev_acc)
        print(
            f"Epoch {ep}: train_loss={train_loss:.3f}  dev_loss={dev_loss:.3f}  dev_acc={dev_acc:.3f}"
        )

    # final test evaluation
    test_acc, _, preds, gts, seqs = evaluate("test")
    swa_val, cwa_val = swa(seqs, gts, preds), cwa(seqs, gts, preds)

    # NRGS
    train_sigs = set(signatures(raw_data["train"]["sequence"]))
    novel = [i for i, sg in enumerate(signatures(seqs)) if sg not in train_sigs]
    NRGS = (sum(1 for i in novel if preds[i] == gts[i]) / len(novel)) if novel else 0.0

    # store results
    experiment_data["embed_dim"][embed_dim] = {
        "losses": losses,
        "metrics": {
            "dev_acc": metrics_dev,
            "test": {"acc": test_acc, "swa": swa_val, "cwa": cwa_val},
            "NRGS": NRGS,
        },
        "predictions": preds,
        "ground_truth": gts,
        "timestamps": str(datetime.datetime.now()),
    }

    # plot loss curve
    plt.figure()
    plt.plot(losses["train"], label="train")
    plt.plot(losses["dev"], label="dev")
    plt.title(f"Loss (embed_dim={embed_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"loss_curve_embed{embed_dim}.png"))
    plt.close()


# ------------------------------------------------------------------ #
# Run experiments across different embedding dimensions
for dim in [32, 64, 96, 128]:
    run_experiment(dim)

# ------------------------------------------------------------------ #
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "experiment_data.json"), "w") as fp:
    json.dump(experiment_data, fp, indent=2)

print("\nFinished hyper-parameter tuning. Results saved to 'working/'.")
