import os, pathlib, random, itertools, time, json
from typing import List, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------- working dir & GPU handling ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------- dataset loading or synthetic fallback ----------
SPR_ROOT = pathlib.Path("./SPR_BENCH")  # adapt if necessary


def fabricate_synthetic(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)

    def _gen_csv(fname, n):
        shapes = list("ABCD")
        colors = list("xyz")
        with open(root / fname, "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                seq = " ".join(
                    random.choice(shapes) + random.choice(colors)
                    for _ in range(random.randint(5, 10))
                )
                label = int(sum(t[0] == "A" for t in seq.split()) % 2)  # simple rule
                f.write(f"{i},{seq},{label}\n")

    _gen_csv("train.csv", 500)
    _gen_csv("dev.csv", 100)
    _gen_csv("test.csv", 100)
    print("Fabricated tiny synthetic SPR_BENCH at", root)


if not SPR_ROOT.exists():
    fabricate_synthetic(SPR_ROOT)

# utilities given in prompt (slightly modified: path param)
from datasets import load_dataset, DatasetDict


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


spr = load_spr_bench(SPR_ROOT)
print({k: len(v) for k, v in spr.items()})

# -------- vocab & tokenisation ----------
PAD = "<pad>"
UNK = "<unk>"


def build_vocab(dataset):
    vocab = {PAD: 0, UNK: 1}
    for seq in dataset["sequence"]:
        for tok in seq.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
id2tok = {i: t for t, i in vocab.items()}


def seq_to_ids(seq: str) -> List[int]:
    return [vocab.get(tok, vocab[UNK]) for tok in seq.strip().split()]


label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
num_classes = len(label2id)


# -------- Dataset/Dataloader ----------
class SPRDataset(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = seq_to_ids(self.seqs[idx])
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(len(ids)),
            "seq_str": self.seqs[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = torch.tensor([b["len"] for b in batch])
    maxlen = lengths.max()
    ids_mat = torch.full((len(batch), maxlen), vocab[PAD], dtype=torch.long)
    for i, b in enumerate(batch):
        ids_mat[i, : b["len"]] = b["ids"]
    labels = torch.stack([b["label"] for b in batch])
    seq_strs = [b["seq_str"] for b in batch]
    return {"ids": ids_mat, "len": lengths, "label": labels, "seq_str": seq_strs}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)


# -------- metrics ----------
def count_color_variety(seq: str):
    return len({tok[1] for tok in seq.split()})


def count_shape_variety(seq: str):
    return len({tok[0] for tok in seq.split()})


def pcwa(seqs, y_true, y_pred):
    weights = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights)


def cwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


def swa(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_true, y_pred)]
    return sum(c) / sum(w)


# -------- model ----------
class SPRClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hid=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD])
        self.gru = nn.GRU(embed_dim, hid, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(hid * 2, num_classes)

    def forward(self, ids, lengths):
        emb = self.embed(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[0], h[1]], dim=-1)  # bi-GRU last hidden
        return self.lin(h)


model = SPRClassifier(len(vocab), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------- experiment data store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

# -------- training loop ----------
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0, 0
    for batch in train_loader:
        batch_ids = batch["ids"].to(device)
        batch_len = batch["len"].to(device)
        batch_lab = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(batch_ids, batch_len)
        loss = criterion(logits, batch_lab)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_ids.size(0)
        n += batch_ids.size(0)
    train_loss = total_loss / n

    # ---- validation ----
    model.eval()
    val_loss, n = 0, 0
    all_preds, all_true, all_seqs = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch_ids = batch["ids"].to(device)
            batch_len = batch["len"].to(device)
            batch_lab = batch["label"].to(device)
            logits = model(batch_ids, batch_len)
            loss = criterion(logits, batch_lab)
            val_loss += loss.item() * batch_ids.size(0)
            n += batch_ids.size(0)

            preds = logits.argmax(-1).cpu().tolist()
            trues = batch_lab.cpu().tolist()
            seqs = batch["seq_str"]
            all_preds.extend(preds)
            all_true.extend(trues)
            all_seqs.extend(seqs)
    val_loss /= n
    acc = np.mean([p == t for p, t in zip(all_preds, all_true)])
    val_pcwa = pcwa(all_seqs, all_true, all_preds)
    val_cwa = cwa(all_seqs, all_true, all_preds)
    val_swa = swa(all_seqs, all_true, all_preds)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"acc={acc:.3f}  PCWA={val_pcwa:.3f}  CWA={val_cwa:.3f}  SWA={val_swa:.3f}"
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        {"acc": acc, "PCWA": val_pcwa, "CWA": val_cwa, "SWA": val_swa}
    )
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

    # keep last epoch preds/gt only
    if epoch == EPOCHS:
        experiment_data["SPR_BENCH"]["predictions"] = all_preds
        experiment_data["SPR_BENCH"]["ground_truth"] = all_true

# -------- confusion matrix plot ----------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(
    experiment_data["SPR_BENCH"]["ground_truth"],
    experiment_data["SPR_BENCH"]["predictions"],
)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix SPR_BENCH")
plt.savefig(os.path.join(working_dir, "spr_confusion.png"))
plt.close()

# -------- save experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
