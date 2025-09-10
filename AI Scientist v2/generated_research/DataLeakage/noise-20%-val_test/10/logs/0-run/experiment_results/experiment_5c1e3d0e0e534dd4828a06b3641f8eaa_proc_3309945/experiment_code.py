import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ----------------- basics & reproducibility -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------- load SPR_BENCH -----------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for sp in ["train", "dev", "test"]:
        d[sp] = _load(f"{sp}.csv")
    return d


spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not spr_root.exists():
    spr_root = pathlib.Path("SPR_BENCH/")
dsets = load_spr_bench(spr_root)


# ----------------- build vocab -----------------
def build_vocab(split):
    charset = set()
    for s in split["sequence"]:
        charset.update(s)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(dsets["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(dsets["train"]["label"]))


# ----------------- dataset & dataloader -----------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_ids = [self.vocab[c] for c in self.seqs[idx]]
        return {
            "seq": torch.tensor(seq_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lens = torch.tensor([len(b["seq"]) for b in batch])
    max_len = lens.max().item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    bag = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = b["seq"].size(0)
        padded[i, :l] = b["seq"]
        bag[i].index_add_(0, b["seq"], torch.ones(l))
    labels = torch.stack([b["label"] for b in batch])
    return {
        "input": padded,
        "lengths": lens,
        "bag": bag,
        "label": labels,
    }


batch_size = 128
train_dl = DataLoader(
    SPRDataset(dsets["train"], vocab), batch_size, shuffle=True, collate_fn=collate
)
dev_dl = DataLoader(
    SPRDataset(dsets["dev"], vocab), batch_size, shuffle=False, collate_fn=collate
)
test_dl = DataLoader(
    SPRDataset(dsets["test"], vocab), batch_size, shuffle=False, collate_fn=collate
)


# ----------------- model definitions -----------------
class AttnUniLSTM(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(hid))
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,L,H)
        attn_scores = torch.tanh(out) @ self.attn_vec  # (B,L)
        mask = x != 0
        attn_scores[~mask] = -1e9
        w = torch.softmax(attn_scores, 1).unsqueeze(-1)  # (B,L,1)
        ctx = (out * w).sum(1)  # (B,H)
        return self.fc(ctx)  # (B,C)


class BagClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.lin = nn.Linear(vocab_sz, num_classes)

    def forward(self, bag):
        return self.lin(bag)


class HybridModel(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.seq_model = AttnUniLSTM(vocab_sz)
        self.bag_model = BagClassifier(vocab_sz)

    def forward(self, inp, lens, bag):
        log_seq = self.seq_model(inp, lens)
        log_bag = self.bag_model(bag)
        return (log_seq + log_bag) / 2, log_bag


model = HybridModel(vocab_size).to(device)
l1_lambda = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------- helpers -----------------
def epoch_pass(dataloader, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in dataloader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, logits_bag = model(batch["input"], batch["lengths"], batch["bag"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
            loss += l1_lambda * model.bag_model.lin.weight.abs().mean()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = total_loss / len(dataloader.dataset)
    macro_f1 = f1_score(gts, preds, average="macro")
    return avg_loss, macro_f1, preds, gts


# ----------------- experiment logs -----------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": [], "Interpretable-Accuracy": None},
        "rules": {},
        "predictions": [],
        "ground_truth": [],
    }
}
rec = experiment_data["SPR_BENCH"]

# ----------------- training loop (bug fixed) -----------------
best_f1, best_state = 0.0, None
epochs = 12
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = epoch_pass(train_dl, train=True)
    val_loss, val_f1, _, _ = epoch_pass(dev_dl, train=False)
    rec["losses"]["train"].append(tr_loss)
    rec["losses"]["val"].append(val_loss)
    rec["metrics"]["train_f1"].append(tr_f1)
    rec["metrics"]["val_f1"].append(val_f1)

    if val_f1 > best_f1:  # -------- BUG FIX HERE --------
        best_f1 = val_f1
        best_state = model.state_dict()

    print(f"Epoch {ep:02d}: val_loss={val_loss:.4f}  val_F1={val_f1:.4f}")

# restore best model
model.load_state_dict(best_state)

# ----------------- rule extraction -----------------
with torch.no_grad():
    w = model.bag_model.lin.weight.detach().cpu().numpy()
rules = {c: int(np.argmax(w[c, 1:]) + 1) for c in range(num_classes)}
rec["rules"] = {c: itos[idx] for c, idx in rules.items()}
print("Extracted one-hot rules per class:", rec["rules"])


def rule_predict(seq):
    for ch in seq:
        vid = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if vid == tok:
                return cls
    return None


# ----------------- interpretable accuracy -----------------
def compute_interpretable_accuracy(hf_split):
    good = 0
    total = len(hf_split)
    model.eval()
    with torch.no_grad():
        for seq, lbl in zip(hf_split["sequence"], hf_split["label"]):
            # rule prediction
            rule_out = rule_predict(seq)
            # model prediction
            seq_ids = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(
                device
            )
            lens = torch.tensor([seq_ids.size(1)], dtype=torch.long).to(device)
            bag = torch.zeros(1, vocab_size, device=device)
            bag[0].index_add_(
                0, seq_ids.squeeze(0), torch.ones(seq_ids.size(1), device=device)
            )
            logit, _ = model(seq_ids, lens, bag)
            mdl_out = int(torch.argmax(logit, 1).item())
            if mdl_out == lbl and rule_out == mdl_out:
                good += 1
    return good / total


interp_acc = compute_interpretable_accuracy(dsets["test"])
rec["metrics"]["Interpretable-Accuracy"] = interp_acc
print(f"Interpretable-Accuracy on TEST: {interp_acc:.4f}")

# ----------------- final test metrics -----------------
test_loss, test_f1, preds_test, gts_test = epoch_pass(test_dl, train=False)
rec["predictions"] = preds_test
rec["ground_truth"] = gts_test
print(f"Hybrid TEST macro-F1 = {test_f1:.4f}")

# ----------------- save everything -----------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
