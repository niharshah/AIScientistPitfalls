import os, pathlib, random, time, numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from collections import Counter

# -----------------------  Book-keeping & device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "weight_decay": {
        "SPR_BENCH": {
            "settings": [],  # list of weight_decay values tried
            "metrics": {"train": [], "val": [], "test": []},  # each item per setting
            "losses": {"train": [], "val": []},  # epoch curves per setting
            "predictions": [],  # test predictions per setting
            "ground_truth": [],  # corresponding ground-truth labels
            "timestamps": [],  # wall-clock time after each run
        }
    }
}

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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
            labels.append(
                int(any(tok[0] == "A" for tok in seq.split()))
            )  # contains shape A
            seqs.append(seq)
        return {"id": list(range(n)), "sequence": seqs, "label": labels}

    spr = {"train": synth(2000), "dev": synth(300), "test": synth(300)}

    def shape_weighted_accuracy(seqs, y_t, y_p):
        return sum(int(t == p) for t, p in zip(y_t, y_p)) / len(y_t)

    color_weighted_accuracy = shape_weighted_accuracy

# -----------------------  Vocabulary ---------------------------------------
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
        self.ids, self.seqs, self.labels = (
            split["id"],
            split["sequence"],
            split["label"],
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "seq": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    lengths = [len(item["seq"]) for item in batch]
    maxlen = max(lengths)
    seqs = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        seqs[i, : lengths[i]] = item["seq"]
    return {
        "seq": seqs,
        "lengths": torch.tensor(lengths),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_seq": [b["raw_seq"] for b in batch],
    }


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRDataset(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate
)


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
EPOCHS = 5
weight_decays = [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


# -----------------------  Helper: evaluation -------------------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_seqs = [], [], []
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch["seq"], batch["lengths"])
            loss_total += criterion(out, batch["label"]).item() * len(batch["label"])
            preds = out.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().tolist())
            all_seqs.extend(batch["raw_seq"])
    avg_loss = loss_total / len(all_labels)
    swa = shape_weighted_accuracy(all_seqs, all_labels, all_preds)
    cwa = color_weighted_accuracy(all_seqs, all_labels, all_preds)
    hwa = 2 * swa * cwa / (swa + cwa + 1e-8)
    return avg_loss, swa, cwa, hwa, all_preds, all_labels


# -----------------------  Hyper-parameter sweep ----------------------------
for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    model = GRUClassifier(len(vocab), 32, 64, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)

    train_losses, val_losses, val_hwas = [], [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(batch["seq"], batch["lengths"])
            loss = criterion(out, batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch["label"])
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, _, _, hwa, _, _ = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_hwas.append(hwa)
        print(
            f"  Epoch {ep}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  HWA={hwa:.4f}"
        )

    # final evaluation on train/val/test
    _, _, _, train_hwa, _, _ = evaluate(model, train_loader)
    val_loss, _, _, val_hwa, _, _ = evaluate(model, val_loader)
    test_loss, swa, cwa, test_hwa, preds, labels = evaluate(model, test_loader)
    # store
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    ed["settings"].append(wd)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["metrics"]["train"].append(train_hwa)
    ed["metrics"]["val"].append(val_hwa)
    ed["metrics"]["test"].append(test_hwa)
    ed["predictions"].append(preds)
    ed["ground_truth"].append(labels)
    ed["timestamps"].append(time.time())
    print(f"  >> Test HWA={test_hwa:.4f} (SWA {swa:.4f} / CWA {cwa:.4f})")

    # optional confusion plot per setting
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(3, 3))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"wd={wd}")
        plt.xlabel("Pred")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"confusion_wd_{wd}.png"))
        plt.close()
    except:
        pass
    torch.cuda.empty_cache()

# -----------------------  Save results -------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy in working_dir.")
