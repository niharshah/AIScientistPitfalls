import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# ----------------------  experiment data dict ------------------------------
experiment_data = {"embedding_dim": {"SPR_BENCH": {}}}

# ----------------------  folders -------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------  GPU / Device -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------  Dataset loading  ---------------------------------
try:
    from SPR import load_spr_bench, shape_weighted_accuracy, color_weighted_accuracy

    DATA_PATH = pathlib.Path(os.getenv("SPR_DATA_PATH", "./SPR_BENCH"))
    spr = load_spr_bench(DATA_PATH)
except Exception:
    print("SPR_BENCH not found, generating synthetic data…")
    shapes, colours = ["A", "B", "C"], ["r", "g", "b"]

    def synth(n):
        seqs, labels = [], []
        for _ in range(n):
            length = random.randint(4, 10)
            seq = " ".join(
                random.choice(shapes) + random.choice(colours) for _ in range(length)
            )
            labels.append(int(any(tok[0] == "A" for tok in seq.split())))
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -----------------------  Vocabulary build  --------------------------------
train_seqs = spr["train"]["sequence"]
counter = Counter(tok for seq in train_seqs for tok in seq.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for tok in counter:
    vocab[tok] = len(vocab)
pad_idx, unk_idx = vocab["<PAD>"], vocab["<UNK>"]
num_classes = len(set(spr["train"]["label"]))


def encode(seq):
    return [vocab.get(tok, unk_idx) for tok in seq.split()]


# -----------------------  Torch Dataset ------------------------------------
class SPRDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids, self.seqs, self.label = split["id"], split["sequence"], split["label"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(b["seq"]) for b in batch]
    maxlen = max(lengths)
    padded = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    return {
        "seq": padded,
        "lengths": torch.tensor(lengths),
        "label": labels,
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_ds, val_ds, test_ds = (SPRDataset(spr[s]) for s in ["train", "dev", "test"])
train_loader = DataLoader(train_ds, 64, True, collate_fn=collate)
val_loader = DataLoader(val_ds, 128, False, collate_fn=collate)
test_loader = DataLoader(test_ds, 128, False, collate_fn=collate)


# -----------------------  Model --------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


criterion = nn.CrossEntropyLoss()


def evaluate(net, loader):
    net.eval()
    tot_loss, preds, labs, raws = 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = net(batch_t["seq"], batch_t["lengths"])
            tot_loss += criterion(out, batch_t["label"]).item() * len(batch_t["label"])
            p = out.argmax(-1).cpu().tolist()
            preds += p
            labs += batch_t["label"].cpu().tolist()
            raws += batch["raw_seq"]
    loss = tot_loss / len(labs)
    swa = shape_weighted_accuracy(raws, labs, preds)
    cwa = color_weighted_accuracy(raws, labs, preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return loss, hwa, preds, labs, raws


# -----------------------  Hyper-param loop ---------------------------------
candidate_dims = [16, 32, 64, 128]
best_val_hwa, best_state, best_dim = -1, None, None

for dim in candidate_dims:
    print(f"\n=== Training with embedding_dim={dim} ===")
    model = GRUClassifier(len(vocab), dim, 64, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 5
    exp_rec = {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(batch_t["seq"], batch_t["lengths"])
            loss = criterion(out, batch_t["label"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_t["label"])
        train_loss = epoch_loss / len(train_ds)
        val_loss, val_hwa, *_ = evaluate(model, val_loader)
        print(f"Dim {dim} | Epoch {epoch}: val_loss {val_loss:.4f} HWA {val_hwa:.4f}")
        exp_rec["losses"]["train"].append(train_loss)
        exp_rec["losses"]["val"].append(val_loss)
        exp_rec["metrics"]["val"].append(val_hwa)
        exp_rec["timestamps"].append(time.time())
    # store results
    experiment_data["embedding_dim"]["SPR_BENCH"][str(dim)] = exp_rec
    # keep best
    if val_hwa > best_val_hwa:
        best_val_hwa, best_dim = val_hwa, dim
        best_state = model.state_dict()
    del model
    torch.cuda.empty_cache()

print(f"\nBest embedding_dim = {best_dim} with validation HWA={best_val_hwa:.4f}")

# -----------------------  Test evaluation ----------------------------------
best_model = GRUClassifier(len(vocab), best_dim, 64, num_classes, pad_idx).to(device)
best_model.load_state_dict(best_state)
test_loss, test_hwa, preds, labels, seqs = evaluate(best_model, test_loader)
print(f"TEST: loss {test_loss:.4f} | HWA {test_hwa:.4f}")

# save test results
best_exp = experiment_data["embedding_dim"]["SPR_BENCH"][str(best_dim)]
best_exp["predictions"] = preds
best_exp["ground_truth"] = labels
best_exp["test_hwa"] = test_hwa

# -----------------------  Confusion matrix ---------------------------------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (dim={best_dim})")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_confusion.png"))
    plt.close()
except:
    pass

# -----------------------  Save experiment data -----------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data to working_dir.")
