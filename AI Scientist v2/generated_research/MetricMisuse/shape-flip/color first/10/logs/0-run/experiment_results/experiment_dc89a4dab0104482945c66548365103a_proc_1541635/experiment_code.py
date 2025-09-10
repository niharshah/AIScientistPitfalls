import os, pathlib, time, numpy as np, torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from datasets import load_dataset, DatasetDict

# -------------------------------------------------------------------
# working dir + device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------
# ---- SPR helpers (copied from given code) --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_color_variety(s) for s in sequences]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(sequences, y_true, y_pred):
    w = [count_shape_variety(s) for s in sequences]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# -------------------------------------------------------------------
# ---- Dataset to Graph ---------------------------------------------
def build_token_mapping(dataset):
    token2id = {}
    for seq in dataset["train"]["sequence"]:
        for tok in seq.split():
            if tok not in token2id:
                token2id[tok] = len(token2id)
    return token2id


def build_label_mapping(dataset):
    lbls = sorted(list(set(dataset["train"]["label"])))
    return {l: i for i, l in enumerate(lbls)}


def seq_to_graph(sequence, label, token2id, label2id):
    tokens = sequence.split()
    x = torch.tensor([token2id[tok] for tok in tokens], dtype=torch.long)
    # chain edges
    if len(tokens) > 1:
        idx = torch.arange(len(tokens) - 1)
        edge_index = torch.vstack(
            [torch.cat([idx, idx + 1]), torch.cat([idx + 1, idx])]
        )
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
    y = torch.tensor([label2id[label]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def convert_split(ds_split, token2id, label2id):
    graphs = []
    for seq, lbl in zip(ds_split["sequence"], ds_split["label"]):
        graphs.append(seq_to_graph(seq, lbl, token2id, label2id))
    return graphs


# -------------------------------------------------------------------
# ---- Model ---------------------------------------------------------
class GCNGraphClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze())  # [N, emb_dim]
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)  # [B, hidden]
        return self.lin(x)


# -------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    gts = []
    seqs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1).cpu().tolist()
        gt = batch.y.cpu().tolist()
        preds.extend(pred)
        gts.extend(gt)
        seqs.extend(
            batch.sequence if hasattr(batch, "sequence") else [""] * batch.num_graphs
        )
    return total_loss / len(loader.dataset), preds, gts, seqs


# -------------------------------------------------------------------
# ---- Main experiment ----------------------------------------------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)

token2id = build_token_mapping(spr)
label2id = build_label_mapping(spr)
id2label = {v: k for k, v in label2id.items()}

train_graphs = convert_split(spr["train"], token2id, label2id)
dev_graphs = convert_split(spr["dev"], token2id, label2id)
test_graphs = convert_split(spr["test"], token2id, label2id)

# keep raw sequences for metrics
for g, seq in zip(train_graphs, spr["train"]["sequence"]):
    g.sequence = seq
for g, seq in zip(dev_graphs, spr["dev"]["sequence"]):
    g.sequence = seq
for g, seq in zip(test_graphs, spr["test"]["sequence"]):
    g.sequence = seq

batch_size = 64
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_graphs, batch_size=batch_size)
test_loader = DataLoader(test_graphs, batch_size=batch_size)

model = GCNGraphClassifier(len(token2id), 32, 64, len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

experiment_data = {
    "SprBench": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_gts, val_seqs = evaluate(model, dev_loader, criterion)

    # convert ids back to labels for metric
    val_preds_lbl = [id2label[p] for p in val_preds]
    val_gts_lbl = [id2label[g] for g in val_gts]
    cwa = color_weighted_accuracy(val_seqs, val_gts_lbl, val_preds_lbl)
    swa = shape_weighted_accuracy(val_seqs, val_gts_lbl, val_preds_lbl)
    hwa = harmonic_weighted_accuracy(cwa, swa)

    experiment_data["SprBench"]["losses"]["train"].append(tr_loss)
    experiment_data["SprBench"]["losses"]["val"].append(val_loss)
    experiment_data["SprBench"]["metrics"]["train"].append(None)  # placeholder
    experiment_data["SprBench"]["metrics"]["val"].append(
        {"CWA": cwa, "SWA": swa, "HWA": hwa, "epoch": epoch}
    )

    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  CWA={cwa:.4f}  SWA={swa:.4f}  HWA={hwa:.4f}"
    )

# Final evaluation on test
_, test_preds, test_gts, test_seqs = evaluate(model, test_loader, criterion)
test_preds_lbl = [id2label[p] for p in test_preds]
test_gts_lbl = [id2label[g] for g in test_gts]
cwa_test = color_weighted_accuracy(test_seqs, test_gts_lbl, test_preds_lbl)
swa_test = shape_weighted_accuracy(test_seqs, test_gts_lbl, test_preds_lbl)
hwa_test = harmonic_weighted_accuracy(cwa_test, swa_test)
experiment_data["SprBench"]["predictions"] = test_preds_lbl
experiment_data["SprBench"]["ground_truth"] = test_gts_lbl
print(f"TEST: CWA={cwa_test:.4f}  SWA={swa_test:.4f}  HWA={hwa_test:.4f}")

# -------------------------------------------------------------------
# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
