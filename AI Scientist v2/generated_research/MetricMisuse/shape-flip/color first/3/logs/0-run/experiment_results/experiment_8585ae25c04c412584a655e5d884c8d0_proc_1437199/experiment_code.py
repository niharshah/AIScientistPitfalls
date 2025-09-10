import os, csv, pathlib, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# ------------------------------------------------------------------
# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# experiment data container
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_BWA": [], "val_BWA": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}


# ------------------------------------------------------------------
# helper: metrics ---------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_color_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


def shape_weighted_accuracy(sequences, y_true, y_pred):
    weights = [count_shape_variety(s) for s in sequences]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


# ------------------------------------------------------------------
# dataset utils -----------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


def maybe_create_synthetic_dataset(root_dir: pathlib.Path):
    if root_dir.exists():
        return
    print("Real dataset not found; creating synthetic SPR_BENCH...")
    root_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": 1000, "dev": 200, "test": 200}
    shapes = list("ABCD")
    colors = list("rgbc")
    for split, nrows in splits.items():
        with open(root_dir / f"{split}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "sequence", "label"])
            for i in range(nrows):
                seq_len = random.randint(4, 10)
                tokens = [
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(seq_len)
                ]
                sequence = " ".join(tokens)
                # simple rule: label 1 if #unique shapes > #unique colours else 0
                label = int(
                    count_shape_variety(sequence) > count_color_variety(sequence)
                )
                writer.writerow([f"{split}_{i}", sequence, label])


# ------------------------------------------------------------------
# build vocab & graphs ---------------------------------------------
def build_vocabs_and_graphs(dataset, label_map=None, shape_map=None, color_map=None):
    if label_map is None:
        label_map = {}
    graphs = []
    for ex in dataset:
        seq = ex["sequence"]
        label = ex["label"]
        tokens = seq.strip().split()
        shape_ids, color_ids = [], []
        for tok in tokens:
            s, c = tok[0], tok[1]
            if shape_map is not None:
                shape_ids.append(shape_map.get(s, 0))
            else:
                if s not in label_map:  # temporarily misuse label_map as collector
                    label_map[s] = None
                shape_ids.append(0)
            if color_map is not None:
                color_ids.append(color_map.get(c, 0))
            else:
                if c not in label_map:
                    label_map[c] = None
                color_ids.append(0)
        # edges (bidirectional)
        edges = []
        for i in range(len(tokens) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(
            shape=torch.tensor(shape_ids, dtype=torch.long),
            color=torch.tensor(color_ids, dtype=torch.long),
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            sequence=seq,
        )
        graphs.append(data)
    return graphs


# ------------------------------------------------------------------
# main workflow -----------------------------------------------------
def main():
    data_path = pathlib.Path(
        os.getenv("SPR_DATA_PATH", os.path.join(os.getcwd(), "SPR_BENCH"))
    )
    maybe_create_synthetic_dataset(data_path)
    datasets_dict = load_spr_bench(data_path)

    # build vocabularies from train
    shapes_set, colors_set, labels_set = set(), set(), set()
    for ex in datasets_dict["train"]:
        seq = ex["sequence"]
        labels_set.add(ex["label"])
        for tok in seq.split():
            shapes_set.add(tok[0])
            colors_set.add(tok[1])
    shape2idx = {s: i for i, s in enumerate(sorted(shapes_set))}
    color2idx = {c: i for i, c in enumerate(sorted(colors_set))}
    label2idx = {l: i for i, l in enumerate(sorted(labels_set))}
    num_shapes, num_colors, num_classes = len(shape2idx), len(color2idx), len(label2idx)

    # remap labels in all splits
    for split in ("train", "dev", "test"):
        datasets_dict[split] = datasets_dict[split].map(
            lambda ex: {"label": label2idx[ex["label"]]}
        )

    # convert to graphs
    train_graphs = build_vocabs_and_graphs(
        datasets_dict["train"], shape_map=shape2idx, color_map=color2idx
    )
    dev_graphs = build_vocabs_and_graphs(
        datasets_dict["dev"], shape_map=shape2idx, color_map=color2idx
    )
    test_graphs = build_vocabs_and_graphs(
        datasets_dict["test"], shape_map=shape2idx, color_map=color2idx
    )
    # loaders
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_graphs, batch_size=128)
    test_loader = DataLoader(test_graphs, batch_size=128)

    # model
    class SPRGNN(nn.Module):
        def __init__(self, n_shapes, n_colors, hidden, n_cls):
            super().__init__()
            self.shape_emb = nn.Embedding(n_shapes, 8)
            self.color_emb = nn.Embedding(n_colors, 8)
            self.lin0 = nn.Linear(16, hidden)
            self.conv1 = GCNConv(hidden, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.classifier = nn.Linear(hidden, n_cls)

        def forward(self, data):
            x = torch.cat(
                [self.shape_emb(data.shape), self.color_emb(data.color)], dim=-1
            )
            x = F.relu(self.lin0(x))
            x = F.relu(self.conv1(x, data.edge_index))
            x = F.relu(self.conv2(x, data.edge_index))
            x = global_mean_pool(x, data.batch)
            return self.classifier(x)

    model = SPRGNN(num_shapes, num_colors, hidden=64, n_cls=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(loader, train=False):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, sequences, y_true, y_pred = 0.0, [], [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).detach().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch.y.view(-1).cpu().tolist())
            sequences.extend(batch.sequence)
        avg_loss = total_loss / len(loader.dataset)
        cwa = color_weighted_accuracy(sequences, y_true, y_pred)
        swa = shape_weighted_accuracy(sequences, y_true, y_pred)
        bwa = (cwa + swa) / 2.0
        return avg_loss, bwa

    # training loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_bwa = run_epoch(train_loader, train=True)
        val_loss, val_bwa = run_epoch(dev_loader, train=False)

        # store
        experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
        experiment_data["SPR_BENCH"]["metrics"]["train_BWA"].append(tr_bwa)
        experiment_data["SPR_BENCH"]["metrics"]["val_BWA"].append(val_bwa)
        experiment_data["SPR_BENCH"]["timestamps"].append(time.time())

        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, val_BWA = {val_bwa:.4f}, "
            f"train_BWA = {tr_bwa:.4f}, time={time.time()-t0:.1f}s"
        )

    # final test evaluation
    _, test_bwa = run_epoch(test_loader, train=False)
    print(f"Test BWA: {test_bwa:.4f}")

    # gather predictions / groundtruth for test set
    model.eval()
    sequences, y_true, y_pred = [], [], []
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch)
        preds = out.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(batch.y.view(-1).cpu().tolist())
        sequences.extend(batch.sequence)

    experiment_data["SPR_BENCH"]["predictions"] = np.array(y_pred)
    experiment_data["SPR_BENCH"]["ground_truth"] = np.array(y_true)

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print("Saved experiment_data to working directory.")


main()
