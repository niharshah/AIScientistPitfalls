import os, random, pathlib, time, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict

# ---------- experiment data dict ----------
experiment_data = {
    "LSTM": {"SPR_BENCH": {}},
    "BOE": {"SPR_BENCH": {}},  # Bag-of-Embeddings ablation
}

# ---------- device & seeds ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
print(f"Using device: {device}")


# ---------- helper metrics ----------
def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)]
    return sum(c) / max(sum(w), 1)


def harmonic_weighted_accuracy(swa, cwa):
    return 2 * swa * cwa / (swa + cwa + 1e-12)


# ---------- data loading ----------
def load_or_create_dataset():
    root = pathlib.Path("SPR_BENCH")
    if root.exists():
        from SPR import load_spr_bench

        return load_spr_bench(root)
    shapes, colors = "ABCD", "abcd"

    def gen_row(_id):
        length = random.randint(4, 9)
        seq = " ".join(
            random.choice(shapes) + random.choice(colors) for _ in range(length)
        )
        return {"id": _id, "sequence": seq, "label": int(length % 2)}

    return DatasetDict(
        {
            "train": HFDataset.from_list([gen_row(i) for i in range(600)]),
            "dev": HFDataset.from_list([gen_row(1_000 + i) for i in range(200)]),
            "test": HFDataset.from_list([gen_row(2_000 + i) for i in range(200)]),
        }
    )


spr = load_or_create_dataset()
print({k: len(v) for k, v in spr.items()})

# ---------- vocabulary ----------
all_text = " ".join(spr["train"]["sequence"])
vocab = sorted(set(all_text.split()))
tok2idx = {tok: i + 2 for i, tok in enumerate(vocab)}
tok2idx["<PAD>"], tok2idx["<UNK>"] = 0, 1
idx2tok = {i: t for t, i in tok2idx.items()}
vocab_size = len(tok2idx)
print("Vocab size:", vocab_size)


def encode_sequence(seq):
    return [tok2idx.get(tok, 1) for tok in seq.strip().split()]


# ---------- torch dataset ----------
class SPRTorchSet(Dataset):
    def __init__(self, hf_split):
        self.seqs, self.labels = hf_split["sequence"], hf_split["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(encode_sequence(self.seqs[idx]), dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw_seq": self.seqs[idx],
        }


def collate(batch):
    xs = [b["x"] for b in batch]
    lens = [len(x) for x in xs]
    maxlen = max(lens)
    xs_pad = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        xs_pad[i, : len(x)] = x
    ys = torch.stack([b["y"] for b in batch])
    raw = [b["raw_seq"] for b in batch]
    return {
        "x": xs_pad.to(device),
        "len": torch.tensor(lens, dtype=torch.long).to(device),
        "y": ys.to(device),
        "raw_seq": raw,
    }


batch_size = 64
train_loader = DataLoader(
    SPRTorchSet(spr["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRTorchSet(spr["dev"]), batch_size=batch_size, shuffle=False, collate_fn=collate
)
n_classes = len(set(spr["train"]["label"]))
print("Num classes:", n_classes)


# ---------- models ----------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab, emb=64, hid=128, n_out=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_out)

    def forward(self, x, lengths):
        em = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            em, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class BagOfEmbClassifier(nn.Module):
    def __init__(self, vocab, emb=64, n_out=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.fc = nn.Linear(emb, n_out)

    def forward(self, x, lengths):
        em = self.emb(x)  # (B, T, E)
        mask = (x != 0).unsqueeze(-1)  # (B, T, 1)
        summed = (em * mask).sum(1)  # (B, E)
        avg = summed / lengths.unsqueeze(1)  # (B, E)
        return self.fc(avg)


# ---------- training routine ----------
def run_training(model_name, num_epochs):
    model_cls = LSTMClassifier if model_name == "LSTM" else BagOfEmbClassifier
    model = model_cls(vocab_size, n_out=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    rec = {
        "losses": {"train": [], "val": []},
        "metrics": {"val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
    }
    for ep in range(1, num_epochs + 1):
        # train
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            out = model(batch["x"], batch["len"])
            loss = criterion(out, batch["y"])
            loss.backward()
            optim.step()
            t_loss += loss.item() * batch["y"].size(0)
        train_loss = t_loss / len(train_loader.dataset)
        # evaluate
        model.eval()
        v_loss = 0.0
        y_true = []
        y_pred = []
        all_seq = []
        with torch.no_grad():
            for batch in dev_loader:
                out = model(batch["x"], batch["len"])
                loss = criterion(out, batch["y"])
                v_loss += loss.item() * batch["y"].size(0)
                preds = torch.argmax(out, 1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(batch["y"].cpu().tolist())
                all_seq.extend(batch["raw_seq"])
        val_loss = v_loss / len(dev_loader.dataset)
        swa = shape_weighted_accuracy(all_seq, y_true, y_pred)
        cwa = color_weighted_accuracy(all_seq, y_true, y_pred)
        hwa = harmonic_weighted_accuracy(swa, cwa)
        # record
        rec["losses"]["train"].append(train_loss)
        rec["losses"]["val"].append(val_loss)
        rec["metrics"]["val"].append(hwa)
        rec["predictions"].append(y_pred)
        rec["ground_truth"].append(y_true)
        rec["timestamps"].append(time.time())
        print(
            f"[{model_name}] Ep {ep}/{num_epochs} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} HWA={hwa:.3f}"
        )
    return rec


# ---------- hyper-parameter sweep ----------
epoch_options = [5, 10, 20, 30]
for model_name in ["LSTM", "BOE"]:
    for e in epoch_options:
        print(f"\n=== {model_name}: training for {e} epochs ===")
        experiment_data[model_name]["SPR_BENCH"][str(e)] = run_training(model_name, e)

# ---------- save ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment_data.npy")
