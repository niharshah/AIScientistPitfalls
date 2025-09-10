import os, pathlib, random, time, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ------------------------ working dir & device ------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------ reproducibility -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------ experiment data store -----------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_loss": [], "val_loss": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# ------------------------ load SPR_BENCH ------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"] = _load("train.csv")
    d["dev"] = _load("dev.csv")
    d["test"] = _load("test.csv")
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------------ vocabulary ----------------------------------
special_tokens = ["<PAD>", "<MASK>"]
chars = set(ch for s in spr["train"]["sequence"] for ch in s)
itos = special_tokens + sorted(chars)
stoi = {ch: i for i, ch in enumerate(itos)}
pad_id, mask_id, vocab_size = stoi["<PAD>"], stoi["<MASK>"], len(itos)
num_classes = len(set(spr["train"]["label"]))
print(f"vocab={vocab_size}, num_classes={num_classes}")


# ------------------------ datasets ------------------------------------
class SPRClassification(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SPRMLM(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = torch.tensor([stoi[c] for c in self.seqs[idx]], dtype=torch.long)
        return {"input_ids": ids}


def collate_cls(batch):
    inputs = [b["input_ids"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return {"input_ids": padded, "label": labels}


def apply_mlm_noise(ids, mlm_prob=0.15):
    """
    ids: LongTensor [L]
    returns noisy_ids, target (-100 for non-masked positions)
    """
    L = ids.size(0)
    probs = torch.rand(L)
    mask = probs < mlm_prob
    if mask.sum() == 0:  # guarantee at least one mask position
        mask[random.randrange(L)] = True
    target = ids.clone()
    target[~mask] = -100  # ignore index

    # 80% replace with <MASK>, 10% random, 10% unchanged
    rand = torch.rand(L)
    repl_mask = mask & (rand < 0.8)
    random_mask = mask & (rand >= 0.8) & (rand < 0.9)
    ids = ids.clone()
    ids[repl_mask] = mask_id
    if random_mask.any():
        ids[random_mask] = torch.randint(2, vocab_size, size=(random_mask.sum(),))
    # 10% remain unchanged
    return ids, target


def collate_mlm(batch):
    seqs = [b["input_ids"] for b in batch]
    noisy, targets = [], []
    for ids in seqs:
        n, t = apply_mlm_noise(ids)
        noisy.append(n)
        targets.append(t)
    noisy = pad_sequence(noisy, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return {"input_ids": noisy, "labels": targets}


train_cls_ds = SPRClassification(spr["train"]["sequence"], spr["train"]["label"])
dev_cls_ds = SPRClassification(spr["dev"]["sequence"], spr["dev"]["label"])
test_cls_ds = SPRClassification(spr["test"]["sequence"], spr["test"]["label"])

mlm_seqs = spr["train"]["sequence"] + spr["dev"]["sequence"]  # self-sup corpora
mlm_ds = SPRMLM(mlm_seqs)

train_mlm_loader = DataLoader(
    mlm_ds, batch_size=256, shuffle=True, collate_fn=collate_mlm
)
train_cls_loader = DataLoader(
    train_cls_ds, batch_size=128, shuffle=True, collate_fn=collate_cls
)
dev_loader = DataLoader(
    dev_cls_ds, batch_size=128, shuffle=False, collate_fn=collate_cls
)
test_loader = DataLoader(
    test_cls_ds, batch_size=128, shuffle=False, collate_fn=collate_cls
)


# ------------------------ model ---------------------------------------
class HybridTransformer(nn.Module):
    def __init__(self, vocab, d_model, nhead, nlayers, n_cls, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.mlm_head = nn.Linear(d_model, vocab, bias=False)
        self.count_proj = nn.Linear(vocab, d_model)
        self.cls_head = nn.Linear(d_model * 2, n_cls)

    def encode(self, x, pad_mask):
        x = self.embed(x)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return h

    def forward_mlm(self, x, pad_mask):
        h = self.encode(x, pad_mask)
        return self.mlm_head(h)

    def forward_cls(self, x, pad_mask):
        h = self.encode(x, pad_mask)  # [B,L,D]
        mask = (~pad_mask).unsqueeze(-1).type_as(h)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool

        # symbol count vector
        one_hot = torch.zeros(x.size(0), vocab_size, device=x.device)
        valid = x.clone()
        valid[pad_mask] = 0  # zero where pad
        one_hot.scatter_add_(1, valid, (~pad_mask).float())
        counts_emb = self.count_proj(one_hot)
        feat = torch.cat([pooled, counts_emb], dim=-1)
        return self.cls_head(feat)


# ------------------------ instantiate ---------------------------------
d_model = 128
model = HybridTransformer(vocab_size, d_model, 4, 3, num_classes, pad_id).to(device)

# ------------------------ pre-training (MLM) --------------------------
mlm_epochs = 5
mlm_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

print("\n===== MLM pre-training =====")
for epoch in range(1, mlm_epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_mlm_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pad_mask = batch["input_ids"] == pad_id
        logits = model.forward_mlm(batch["input_ids"], pad_mask)
        loss = mlm_criterion(logits.view(-1, vocab_size), batch["labels"].view(-1))
        mlm_optimizer.zero_grad()
        loss.backward()
        mlm_optimizer.step()
        running_loss += loss.item() * batch["input_ids"].size(0)
    avg_loss = running_loss / len(train_mlm_loader.dataset)
    print(f"MLM E{epoch}: loss={avg_loss:.4f}")

# ------------------------ fine-tuning (classification) ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
max_epochs, patience = 30, 6
best_f1, no_improve = 0.0, 0


def evaluate(loader):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pad_mask = batch["input_ids"] == pad_id
            logits = model.forward_cls(batch["input_ids"], pad_mask)
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


print("\n===== Classification fine-tuning =====")
for epoch in range(1, max_epochs + 1):
    model.train()
    run_loss = 0.0
    for batch in train_cls_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pad_mask = batch["input_ids"] == pad_id
        logits = model.forward_cls(batch["input_ids"], pad_mask)
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * batch["label"].size(0)
    train_loss = run_loss / len(train_cls_loader.dataset)

    val_loss, val_f1, _, _ = evaluate(dev_loader)
    print(
        f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
    )

    ed = experiment_data["SPR_BENCH"]
    ed["metrics"]["train_loss"].append(train_loss)
    ed["metrics"]["val_loss"].append(val_loss)
    ed["metrics"]["val_f1"].append(val_f1)
    ed["epochs"].append(epoch)

    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# ------------------------ test evaluation -----------------------------
model.load_state_dict(best_state)
test_loss, test_f1, test_preds, test_gts = evaluate(test_loader)
print(f"\nTEST  : loss={test_loss:.4f}  Macro-F1={test_f1:.4f}")

ed["predictions"], ed["ground_truth"] = test_preds, test_gts

# ------------------------ save artefacts ------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
