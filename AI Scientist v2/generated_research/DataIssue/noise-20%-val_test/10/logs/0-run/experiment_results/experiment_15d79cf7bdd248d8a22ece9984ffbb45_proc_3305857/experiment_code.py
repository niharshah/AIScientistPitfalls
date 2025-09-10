import os, pathlib, numpy as np, torch, math, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score

# ---------- misc / reproducibility ----------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not data_root.exists():
    data_root = pathlib.Path("SPR_BENCH/")  # fallback for local testing
dset = load_spr_bench(data_root)
print({k: len(v) for k, v in dset.items()})


# ---------- vocab ----------
def build_vocab(split):
    chars = set()
    for s in split["sequence"]:
        chars.update(s)
    stoi = {c: i + 1 for i, c in enumerate(sorted(chars))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(dset["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(dset["train"]["label"]))
print("vocab_size", vocab_size, "num_classes", num_classes)


# ---------- torch dataset ----------
class SPRTorch(Dataset):
    def __init__(self, split, vocab):
        self.seqs = split["sequence"]
        self.labels = split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = [self.vocab[c] for c in self.seqs[idx]]
        return {
            "input": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    maxlen = max(lens)
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : len(b["input"])] = b["input"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": torch.tensor(lens), "label": labels}


bs = 256
train_dl = DataLoader(
    SPRTorch(dset["train"], vocab), bs, shuffle=True, collate_fn=collate
)
dev_dl = DataLoader(SPRTorch(dset["dev"], vocab), bs, shuffle=False, collate_fn=collate)
test_dl = DataLoader(
    SPRTorch(dset["test"], vocab), bs, shuffle=False, collate_fn=collate
)


# ---------- Neural Rule Dictionary model ----------
class NRDModel(nn.Module):
    """
    1. Char embeddings -> biGRU pooled feature
    2. Compare to K learnable rule vectors (dot product)
    3. Winning rule id is mapped to class via a trainable lookup table.
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, K_rules, n_cls):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.rule_vecs = nn.Parameter(torch.randn(K_rules, hid_dim * 2))
        self.rule_to_class = nn.Linear(K_rules, n_cls, bias=False)  # soft mapping

    def encode(self, x, lens):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1)
        feat = (out * mask).sum(1) / lens.unsqueeze(1).to(out.dtype)
        return feat  # [B,2H]

    def forward(self, x, lens, return_rule=False):
        feat = self.encode(x, lens)  # [B,2H]
        sims = torch.matmul(feat, self.rule_vecs.t())  # [B,K]
        logits = self.rule_to_class(sims)  # [B,C]
        if return_rule:
            rule_id = sims.argmax(1)  # index of max rule
            return logits, rule_id
        return logits


# ---------- training helpers ----------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dloader, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss = 0
    preds = []
    labels = []
    rules = []
    for batch in dloader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits, rule_id = model(batch["input"], batch["lengths"], return_rule=True)
        loss = criterion(logits, batch["label"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
        rules.extend(rule_id.detach().cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return tot_loss / len(dloader.dataset), acc, f1, preds, labels, rules


# ---------- experiment store ----------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},  # each item=(epoch, acc, f1, rea)
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rules": [],
    }
}

# ---------- training loop ----------
K_rules = 64
model = NRDModel(
    vocab_size, emb_dim=64, hid_dim=128, K_rules=K_rules, n_cls=num_classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
best_val_f1 = 0
best_state = None

for ep in range(1, epochs + 1):
    tr_loss, tr_acc, tr_f1, *_ = run_epoch(model, train_dl, optimizer)
    val_loss, val_acc, val_f1, *_ = run_epoch(model, dev_dl)
    # REA == accuracy because final decision is via rules
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((ep, tr_acc, tr_f1, tr_acc))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(
        (ep, val_acc, val_f1, val_acc)
    )
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
    if ep % 2 == 0 or ep == 1:
        print(
            f"Epoch {ep}: validation_loss = {val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
        )

# ---------- load best and evaluate ----------
model.load_state_dict(best_state)
test_loss, test_acc, test_f1, preds, gts, rules = run_epoch(model, test_dl)
print(
    f"Test Acc={test_acc:.4f}  Test Macro-F1={test_f1:.4f}  (Rule Extraction Accuracy={test_acc:.4f})"
)

experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
experiment_data["SPR_BENCH"]["rules"] = rules
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
