import os, pathlib, time, math, itertools, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# --------------------------- boiler-plate ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": [], "test": {}},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
}
exp_log = experiment_data["SPR_BENCH"]


# --------------------------- helper functions ------------------------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    num = sum(wi if t == p else 0 for wi, t, p in zip(w, y_t, y_p))
    den = max(sum(w), 1)
    return num / den


def rule_signature(sequence: str) -> str:
    return " ".join(tok[0] for tok in sequence.strip().split() if tok)


# --------------------------- data loading ----------------------------
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

# --------------------------- vocab -----------------------------------
PAD, UNK = "<PAD>", "<UNK>"


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    tokens = set(itertools.chain.from_iterable(seq.strip().split() for seq in dataset))
    for tok in sorted(tokens):
        vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
print(f"Vocab size = {vocab_size}")


def encode(seq: str):
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


label_set = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(label_set)}
id2lab = {i: l for l, i in lab2id.items()}


# --------------------------- torch dataset ---------------------------
class SPRTorchDS(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = [lab2id[l] for l in split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "input_ids": torch.tensor(encode(seq), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "shape_cnt": torch.tensor(count_shape_variety(seq), dtype=torch.float),
            "color_cnt": torch.tensor(count_color_variety(seq), dtype=torch.float),
            "raw_seq": seq,
        }


def collate(batch):
    ids = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    shape = torch.stack([b["shape_cnt"] for b in batch]).unsqueeze(-1)
    color = torch.stack([b["color_cnt"] for b in batch]).unsqueeze(-1)
    padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=vocab[PAD])
    return {
        "input_ids": padded,
        "labels": labels,
        "aux": torch.cat([shape, color], dim=1),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_ds, dev_ds, test_ds = (
    SPRTorchDS(spr["train"]),
    SPRTorchDS(spr["dev"]),
    SPRTorchDS(spr["test"]),
)
train_loader = DataLoader(train_ds, 128, shuffle=True, collate_fn=collate)
dev_loader = DataLoader(dev_ds, 256, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_ds, 256, shuffle=False, collate_fn=collate)


# --------------------------- model -----------------------------------
class HybridClassifier(nn.Module):
    def __init__(self, vocab_sz, emb=32, hid=64, num_lab=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.aux_linear = nn.Linear(2, hid)
        self.fc = nn.Linear(hid * 2, num_lab)

    def forward(self, input_ids, aux_feats):
        x = self.embedding(input_ids)
        _, h = self.gru(x)
        aux = torch.relu(self.aux_linear(aux_feats))
        out = torch.cat([h.squeeze(0), aux], dim=-1)
        return self.fc(out)


model = HybridClassifier(vocab_size, 32, 64, len(label_set)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------- symbolic memory (majority vote) ---------------
symbolic_map = {}
for seq, lab in zip(spr["train"]["sequence"], spr["train"]["label"]):
    sig = rule_signature(seq)
    symbolic_map.setdefault(sig, []).append(lab)
symbolic_map = {s: max(labs, key=labs.count) for s, labs in symbolic_map.items()}
print(f"Symbolic memory size: {len(symbolic_map)}")


# --------------------------- evaluation ------------------------------
def evaluate(loader):
    model.eval()
    all_seq, all_true, all_pred = [], [], []
    loss_sum = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            aux = batch["aux"].to(device)
            labs = batch["labels"].to(device)
            logits = model(inp, aux)
            loss = criterion(logits, labs)
            loss_sum += loss.item() * len(labs)
            nn_pred = logits.argmax(dim=-1).cpu().tolist()
            for seq, lp, npred in zip(batch["raw_seq"], labs.cpu().tolist(), nn_pred):
                final_pred = (
                    lab2id[symbolic_map[rule_signature(seq)]]
                    if rule_signature(seq) in symbolic_map
                    else npred
                )
                all_seq.append(seq)
                all_true.append(lp)
                all_pred.append(final_pred)
            total += len(labs)
    swa = shape_weighted_accuracy(all_seq, all_true, all_pred)
    return loss_sum / total, swa, all_pred, all_true, all_seq


# --------------------------- training loop ---------------------------
MAX_EPOCHS, PATIENCE = 15, 3
best_val_loss = float("inf")
patience_ctr = 0
best_state = None
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["aux"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
    train_loss = running_loss / len(train_ds)
    val_loss, val_swa, *_ = evaluate(dev_loader)

    exp_log["losses"]["train"].append(train_loss)
    exp_log["losses"]["val"].append(val_loss)
    exp_log["metrics"]["train"].append({"epoch": epoch})
    exp_log["metrics"]["val"].append({"epoch": epoch, "swa": val_swa})
    exp_log["timestamps"].append(time.time())

    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  SWA={val_swa:.3f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping triggered")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------- final test ------------------------------
test_loss, test_swa, preds, gts, seqs = evaluate(test_loader)
print(f"\nTEST  loss={test_loss:.4f}  SWA={test_swa:.3f}")

exp_log["predictions"] = preds
exp_log["ground_truth"] = gts
exp_log["metrics"]["test"] = {"loss": test_loss, "swa": test_swa}

# --------------------------- save artefacts --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")

# quick plot
fig, ax = plt.subplots()
ax.bar(["SWA"], [test_swa], color="orange")
ax.set_ylim(0, 1)
ax.set_title("Hybrid SPR Test SWA")
plt.tight_layout()
plot_path = os.path.join(working_dir, "spr_hybrid_swa.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
