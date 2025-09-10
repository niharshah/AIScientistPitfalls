import os, pathlib, random, math, time, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, matthews_corrcoef
import matplotlib.pyplot as plt

# ---------- working dir & device ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- SPR data ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = _ld("train.csv"), _ld("dev.csv"), _ld("test.csv")
    return d


def get_spr() -> DatasetDict:
    for p in [
        pathlib.Path("./SPR_BENCH"),
        pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
    ]:
        if (p / "train.csv").exists():
            print("Loading real SPR_BENCH from", p)
            return load_spr_bench(p)
    print("SPR_BENCH not found – creating synthetic toy data")

    def synth(n):
        rows, shapes = "ABCD"
        data = []
        for i in range(n):
            seq = "".join(random.choices(shapes, k=random.randint(5, 15)))
            lbl = int(seq.count("A") % 2 == 0 and seq[-1] in "BC")
            data.append({"id": i, "sequence": seq, "label": lbl})
        return load_dataset(
            "json", data_files={"data": data}, field="data", split="train"
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = synth(4000), synth(1000), synth(1000)
    return d


spr = get_spr()

# ---------- vocab / encoding ----------
all_text = "".join(spr["train"]["sequence"])
vocab = sorted(set(all_text))
stoi = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 = PAD
itos = {i: ch for i, ch in enumerate(["<PAD>"] + vocab)}
vocab_size = len(stoi) + 1
max_len = min(120, max(map(len, spr["train"]["sequence"])))


def encode(seq: str):
    ids = [stoi.get(ch, 0) for ch in seq[:max_len]]
    return ids + [0] * (max_len - len(ids))


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seq, self.lbl = split["sequence"], split["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seq[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.lbl[idx]), dtype=torch.float),
        }


def make_loader(name, bs=128, shuffle=False, max_items=None):
    ds = SPRDataset(spr[name])
    if max_items and len(ds) > max_items:
        idx = torch.randperm(len(ds))[:max_items]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


train_loader = lambda: make_loader("train", shuffle=True, max_items=10000)
dev_loader = lambda: make_loader("dev", shuffle=False, max_items=2000)
test_loader = lambda: make_loader("test", shuffle=False)


# ---------- positional encoding ----------
def positional_encoding(seq_len, d_model, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(0, seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ---------- models ----------
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer(
            "pe", positional_encoding(max_len, d_model, torch.device("cpu"))
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pe[:, : x.size(1), :].to(x.device)
        h = self.encoder(h, src_key_padding_mask=mask)
        lengths = (~mask).sum(1).clamp(min=1).unsqueeze(1)
        pooled = h.masked_fill(mask.unsqueeze(2), 0.0).sum(1) / lengths
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(1)


class BagOfEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        mask = x != 0
        h = self.drop(self.emb(x))
        lengths = mask.sum(1).clamp(min=1).unsqueeze(1)
        pooled = (h * mask.unsqueeze(2)).sum(1) / lengths
        return self.fc(pooled).squeeze(1)


# ---------- experiment container ----------
experiment_data = {
    "transformer": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    },
    "bag_of_embeddings": {
        "metrics": {"train_MCC": [], "val_MCC": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    },
}

criterion = nn.BCEWithLogitsLoss()
dropouts = [0.1, 0.3]
epochs = 6


def run_experiment(tag, ModelClass):
    best_dev, best_state, best_dp = -1, None, None
    for dp in dropouts:
        print(f"\n=== {tag} | dropout {dp} ===")
        model = ModelClass(vocab_size, d_model=128, dropout=dp).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(1, epochs + 1):
            # train
            model.train()
            tr_losses, tr_preds, tr_labels = [], [], []
            for batch in train_loader():
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                logit = model(batch["input_ids"])
                loss = criterion(logit, batch["label"])
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())
                tr_preds.extend((torch.sigmoid(logit) > 0.5).cpu().numpy())
                tr_labels.extend(batch["label"].cpu().numpy())
            train_mcc = matthews_corrcoef(tr_labels, tr_preds)

            # val
            model.eval()
            v_losses, v_preds, v_labels = [], [], []
            with torch.no_grad():
                for batch in dev_loader():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logit = model(batch["input_ids"])
                    v_losses.append(criterion(logit, batch["label"]).item())
                    v_preds.extend((torch.sigmoid(logit) > 0.5).cpu().numpy())
                    v_labels.extend(batch["label"].cpu().numpy())
            val_mcc = matthews_corrcoef(v_labels, v_preds)
            print(
                f"Epoch {ep}: val_loss={np.mean(v_losses):.4f} | train_MCC={train_mcc:.3f} val_MCC={val_mcc:.3f}"
            )

            # record
            experiment_data[tag]["metrics"]["train_MCC"].append(train_mcc)
            experiment_data[tag]["metrics"]["val_MCC"].append(val_mcc)
            experiment_data[tag]["losses"]["train"].append(np.mean(tr_losses))
            experiment_data[tag]["losses"]["val"].append(np.mean(v_losses))
            experiment_data[tag]["epochs"].append((dp, ep))

            if val_mcc > best_dev:
                best_dev, val_losses_best = val_mcc, np.mean(v_losses)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_dp = dp
    # test
    print(f"\nBest dev MCC for {tag} = {best_dev:.3f} (dropout={best_dp})")
    best_model = ModelClass(vocab_size, d_model=128, dropout=best_dp).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    t_preds, t_labels = [], []
    with torch.no_grad():
        for batch in test_loader():
            batch = {k: v.to(device) for k, v in batch.items()}
            logit = best_model(batch["input_ids"])
            t_preds.extend((torch.sigmoid(logit) > 0.5).cpu().numpy())
            t_labels.extend(batch["label"].cpu().numpy())
    test_mcc = matthews_corrcoef(t_labels, t_preds)
    test_f1 = f1_score(t_labels, t_preds, average="macro")
    print(f"{tag} | Test MCC={test_mcc:.3f} | Test F1={test_f1:.3f}")
    experiment_data[tag]["predictions"] = t_preds
    experiment_data[tag]["ground_truth"] = t_labels
    experiment_data[tag]["test_MCC"] = test_mcc
    experiment_data[tag]["test_F1"] = test_f1

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(experiment_data[tag]["losses"]["train"], label="train")
    plt.plot(experiment_data[tag]["losses"]["val"], label="val")
    plt.xlabel("update (epochs aggregated)")
    plt.ylabel("BCE loss")
    plt.legend()
    plt.title(f"Loss curve {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"loss_curve_{tag}.png"))
    plt.close()


run_experiment("transformer", CharTransformer)
run_experiment("bag_of_embeddings", BagOfEmbeddings)

# -------- save all ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("All experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
