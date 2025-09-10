import os, pathlib, numpy as np, torch, random, math, time, json, matplotlib
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- global experiment store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ---------- Dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ("train", "dev", "test"):
        dset[split] = _load(f"{split}.csv")
    return dset


def resolve_spr_path() -> pathlib.Path:
    # 1) environment variable
    env_path = os.environ.get("SPR_BENCH_DIR")
    if env_path and pathlib.Path(env_path).exists():
        return pathlib.Path(env_path)
    # 2) default absolute path used in proposal
    default_abs = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
    if default_abs.exists():
        return default_abs
    # 3) relative path next to current working dir
    local_rel = pathlib.Path(os.getcwd()) / "SPR_BENCH"
    if local_rel.exists():
        return local_rel
    raise FileNotFoundError(
        "Could not locate SPR_BENCH dataset folder. "
        "Set environment variable SPR_BENCH_DIR or place the folder in one of the tried locations."
    )


data_path = resolve_spr_path()
print(f"Loading SPR_BENCH from: {data_path}")
spr = load_spr_bench(data_path)
print({k: len(v) for k, v in spr.items()})

# ---------- Vocabulary ----------
PAD, UNK = "<PAD>", "<UNK>"


def tokenize(seq: str):
    seq = seq.strip()
    return seq.split(" ") if " " in seq else list(seq)


vocab = {PAD: 0, UNK: 1}
for ex in spr["train"]["sequence"]:
    for tok in tokenize(ex):
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq):
    return [vocab.get(tok, vocab[UNK]) for tok in tokenize(seq)]


label_set = sorted(set(spr["train"]["label"]))
label2id = {lbl: i for i, lbl in enumerate(label_set)}
num_labels = len(label2id)
print("Labels:", label2id)


# ---------- Torch Dataset ----------
class SPRTorch(Dataset):
    def __init__(self, hf_split):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(encode(self.seqs[idx]), dtype=torch.long),
            "labels": torch.tensor(label2id[self.labels[idx]], dtype=torch.long),
        }


def collate(batch):
    seqs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    maxlen = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), maxlen), 0, dtype=torch.long)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
        mask[i, : s.size(0)] = True
    return {"input_ids": padded, "attention_mask": mask, "labels": labels}


train_loader = DataLoader(
    SPRTorch(spr["train"]), batch_size=64, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    SPRTorch(spr["dev"]), batch_size=128, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    SPRTorch(spr["test"]), batch_size=128, shuffle=False, collate_fn=collate
)


# ---------- Model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=2, num_cls=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.classifier = nn.Linear(d_model, num_cls)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


model = SimpleTransformer(vocab_size, num_cls=num_labels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- Training & Evaluation ----------
def evaluate(loader):
    model.eval()
    total, correct, loss_tot = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            preds = logits.argmax(-1)
            total += preds.size(0)
            correct += (preds == batch["labels"]).sum().item()
            loss_tot += loss.item() * preds.size(0)
    return loss_tot / total, correct / total


epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    tot_loss, tot_correct, tot_seen = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        preds = logits.argmax(-1)
        tot_seen += preds.size(0)
        tot_correct += (preds == batch["labels"]).sum().item()
        tot_loss += loss.item() * preds.size(0)
    train_loss = tot_loss / tot_seen
    train_acc = tot_correct / tot_seen

    val_loss, val_acc = evaluate(val_loader)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | validation_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    experiment_data["SPR_BENCH"]["metrics"]["train_acc"].append(train_acc)
    experiment_data["SPR_BENCH"]["metrics"]["val_acc"].append(val_acc)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)

# ---------- Test evaluation ----------
test_loss, test_acc = evaluate(test_loader)
print(f"Test accuracy: {test_acc:.4f}")

# collect predictions/ground_truth
model.eval()
all_preds, all_truth = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds = logits.argmax(-1).cpu().numpy().tolist()
        truth = batch["labels"].cpu().numpy().tolist()
        all_preds.extend(preds)
        all_truth.extend(truth)

experiment_data["SPR_BENCH"]["predictions"] = np.array(all_preds)
experiment_data["SPR_BENCH"]["ground_truth"] = np.array(all_truth)

# ---------- Save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
