import os, pathlib, time, gc, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List
from datasets import DatasetDict, load_dataset

# ------------------------------------------------------------------#
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------#
def locate_spr_bench() -> pathlib.Path:
    candidates = [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("../SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
        pathlib.Path(os.getenv("SPR_DATA_PATH", "")),
    ]
    for p in candidates:
        if p and (p / "train.csv").exists() and (p / "dev.csv").exists():
            print(f"Found SPR_BENCH at: {p.resolve()}")
            return p.resolve()
    raise FileNotFoundError("SPR_BENCH dataset not found.")


# ------------------------------------------------------------------#
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


def extract_tokens(seq: str) -> List[str]:
    return seq.strip().split()


def count_color_variety(sequence: str) -> int:
    return len(set(t[1] for t in sequence.strip().split() if len(t) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(t[0] for t in sequence.strip().split() if t))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if yt == yp else 0 for wt, yt, yp in zip(w, y_true, y_pred)]
    return sum(corr) / (sum(w) if sum(w) else 1)


# ------------------------------------------------------------------#
DATA_PATH = locate_spr_bench()
spr = load_spr_bench(DATA_PATH)

# vocab
token_set, label_set = set(), set()
for ex in spr["train"]:
    token_set.update(extract_tokens(ex["sequence"]))
    label_set.add(ex["label"])
token2idx = {tok: i + 1 for i, tok in enumerate(sorted(token_set))}
label2idx = {lab: i for i, lab in enumerate(sorted(label_set))}
idx2label = {i: lab for lab, i in label2idx.items()}


# graphs
def seq_to_data(example):
    tokens = extract_tokens(example["sequence"])
    x = torch.tensor([token2idx.get(t, 0) for t in tokens], dtype=torch.long).unsqueeze(
        -1
    )
    if len(tokens) > 1:
        src = torch.arange(len(tokens) - 1, dtype=torch.long)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([label2idx[example["label"]]], dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, y=y)
    d.seq = example["sequence"]
    return d


train_graphs = [seq_to_data(ex) for ex in spr["train"]]
dev_graphs = [seq_to_data(ex) for ex in spr["dev"]]
test_graphs = [seq_to_data(ex) for ex in spr["test"]]


# ------------------------------------------------------------------#
class SPR_GCN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gcn1 = GCNConv(embed_dim, 64)
        self.gcn2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.embed(x.squeeze(-1))
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.drop(x)
        return self.lin(x)


criterion = nn.CrossEntropyLoss()


def evaluate(model, loader):
    model.eval()
    total_loss, preds, labels, seqs = 0.0, [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.batch)
            loss = criterion(out, b.y.view(-1))
            total_loss += loss.item() * b.num_graphs
            preds.extend(out.argmax(-1).cpu().tolist())
            labels.extend(b.y.view(-1).cpu().tolist())
            seqs.extend(b.seq)
    cwa = color_weighted_accuracy(seqs, labels, preds)
    swa = shape_weighted_accuracy(seqs, labels, preds)
    bwa = (cwa + swa) / 2
    return total_loss / len(loader.dataset), bwa, cwa, swa, preds, labels


# ------------------------------------------------------------------#
experiment_data = {"batch_size": {"SPR_BENCH": {}}}

batch_sizes = [32, 64, 128]
num_epochs = 5
colors = {32: "r", 64: "g", 128: "b"}
plt.figure()

for bs in batch_sizes:
    print(f"\n===== Training with batch_size={bs} =====")
    tr_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    dv_loader = DataLoader(dev_graphs, batch_size=bs, shuffle=False)
    te_loader = DataLoader(test_graphs, batch_size=bs, shuffle=False)

    model = SPR_GCN(len(token2idx), 32, len(label2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * batch.num_graphs
        train_loss = ep_loss / len(tr_loader.dataset)

        val_loss, val_bwa, val_cwa, val_swa, _, _ = evaluate(model, dv_loader)
        _, train_bwa, _, _, _, _ = evaluate(model, tr_loader)

        run_data["losses"]["train"].append(train_loss)
        run_data["losses"]["val"].append(val_loss)
        run_data["metrics"]["train"].append(train_bwa)
        run_data["metrics"]["val"].append(val_bwa)
        run_data["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}/{num_epochs} bs={bs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"BWA={val_bwa:.4f} (CWA={val_cwa:.4f}, SWA={val_swa:.4f})"
        )

    # test evaluation
    test_loss, test_bwa, test_cwa, test_swa, preds, labels = evaluate(model, te_loader)
    run_data["predictions"] = preds
    run_data["ground_truth"] = labels
    run_data["test_metrics"] = {
        "loss": test_loss,
        "bwa": test_bwa,
        "cwa": test_cwa,
        "swa": test_swa,
    }
    print(
        f"Test bs={bs} -> loss:{test_loss:.4f} BWA:{test_bwa:.4f} "
        f"CWA:{test_cwa:.4f} SWA:{test_swa:.4f}"
    )

    # store run_data
    experiment_data["batch_size"]["SPR_BENCH"][str(bs)] = run_data

    epochs = np.arange(1, num_epochs + 1)
    plt.plot(epochs, run_data["metrics"]["val"], label=f"bs={bs}", color=colors[bs])

    # cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

plt.xlabel("Epoch")
plt.ylabel("Dev BWA")
plt.title("Batch size tuning on SPR_BENCH")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(working_dir, "bwa_batchsize_curve.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("experiment_data.npy saved.")
