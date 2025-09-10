# num_epochs hyper-parameter tuning – single-file script
import os, pathlib, random, string, numpy as np, torch, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------- misc ----------------------------
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
print("Running on", device)


# -------------------- load / synth dataset -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        {
            "train": _load("train.csv"),
            "dev": _load("dev.csv"),
            "test": _load("test.csv"),
        }
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if DATA_PATH.exists():
    spr = load_spr_bench(DATA_PATH)
else:
    print("Real SPR_BENCH not found, creating synthetic data.")

    def synth(n):
        rows = []
        for i in range(n):
            seq = "".join(
                random.choices(string.ascii_uppercase[:10], k=random.randint(5, 15))
            )
            rows.append(
                {"id": i, "sequence": seq, "label": int(seq.count("A") % 2 == 0)}
            )
        return rows

    def to_ds(rows):
        return load_dataset("json", data_files={"train": [rows]}, split="train")

    spr = DatasetDict()
    spr["train"] = to_ds(synth(2000))
    spr["dev"] = to_ds(synth(400))
    spr["test"] = to_ds(synth(400))
print({k: len(v) for k, v in spr.items()})

# ---------------------- vocab + encode ---------------------
vocab = {"<pad>": 0, "<unk>": 1}
for ex in spr["train"]:
    for ch in ex["sequence"]:
        vocab.setdefault(ch, len(vocab))
vocab_size = len(vocab)
max_len = min(max(len(ex["sequence"]) for ex in spr["train"]), 120)


def encode(seq):
    ids = [vocab.get(ch, 1) for ch in seq][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


class SPRTorchDataset(Dataset):
    def __init__(self, hfds):
        self.hfds = hfds

    def __len__(self):
        return len(self.hfds)

    def __getitem__(self, idx):
        ex = self.hfds[idx]
        return {
            "input_ids": torch.tensor(encode(ex["sequence"]), dtype=torch.long),
            "label": torch.tensor(int(ex["label"]), dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


train_ds, dev_ds = SPRTorchDataset(spr["train"]), SPRTorchDataset(spr["dev"])
train_loader = lambda bs: DataLoader(
    train_ds, batch_size=bs, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=collate)


# ------------------------- model ---------------------------
class CharGRU(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128, cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, cls)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.fc(h.squeeze(0))


# ------------------- training procedure -------------------
def train_for(max_epochs, patience=3):
    model = CharGRU(vocab_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    tr_losses, val_losses, tr_f1s, val_f1s = [], [], [], []
    best_f1, best_pred, best_label = -1, None, None
    no_up = 0
    for ep in range(1, max_epochs + 1):
        # --- train
        model.train()
        tot_loss = tot_items = 0
        for batch in train_loader(128):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch["input_ids"])
            loss = crit(logits, batch["labels"])
            loss.backward()
            optim.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            tot_items += batch["labels"].size(0)
        tr_losses.append(tot_loss / tot_items)

        # --- eval
        model.eval()
        v_loss = v_items = 0
        preds, labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                log = model(batch["input_ids"])
                loss = crit(log, batch["labels"])
                v_loss += loss.item() * batch["labels"].size(0)
                v_items += batch["labels"].size(0)
                preds.extend(log.argmax(1).cpu().tolist())
                labels.extend(batch["labels"].cpu().tolist())
        val_losses.append(v_loss / v_items)
        macro = f1_score(labels, preds, average="macro")
        val_f1s.append(macro)

        # collect train f1 quickly
        tr_preds, tr_labels = [], []
        for batch in train_loader(512):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                tr_logits = model(batch["input_ids"])
            tr_preds.extend(tr_logits.argmax(1).cpu().tolist())
            tr_labels.extend(batch["labels"].cpu().tolist())
        tr_f1s.append(f1_score(tr_labels, tr_preds, average="macro"))

        print(
            f"[{max_epochs}-ep run] epoch {ep}: val_loss={val_losses[-1]:.4f}, val_F1={macro:.4f}"
        )

        # early stop
        if macro > best_f1:
            best_f1, best_pred, best_label = macro, preds.copy(), labels.copy()
            no_up = 0
        else:
            no_up += 1
        if no_up >= patience:
            print("Early stopping.")
            break
    return {
        "metrics": {"train": tr_f1s, "val": val_f1s},
        "losses": {"train": tr_losses, "val": val_losses},
        "predictions": best_pred,
        "ground_truth": best_label,
    }


# ------------------- hyper-param search --------------------
experiment_data = {"num_epochs": {"SPR_BENCH": {}}}
for ep_budget in [5, 10, 15, 20]:
    result = train_for(ep_budget, patience=3)
    experiment_data["num_epochs"]["SPR_BENCH"][f"epochs_{ep_budget}"] = result

# ----------------------- save ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved to", os.path.join(working_dir, "experiment_data.npy"))
