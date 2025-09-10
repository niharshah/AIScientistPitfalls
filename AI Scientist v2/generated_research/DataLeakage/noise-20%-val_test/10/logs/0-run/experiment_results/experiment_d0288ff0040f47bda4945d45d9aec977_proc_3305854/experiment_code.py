import os, pathlib, torch, numpy as np, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# --------------------------------------------------------------------- #
# working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# --------------------------------------------------------------------- #
# data utilities
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(f"{s}.csv")
    return d


def build_vocab(split):
    charset = set()
    for seq in split["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


class SPRSet(Dataset):
    def __init__(self, hf_ds, vocab):
        self.seq = hf_ds["sequence"]
        self.lab = hf_ds["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(
                [self.vocab[c] for c in self.seq[idx]], dtype=torch.long
            ),
            "label": torch.tensor(self.lab[idx], dtype=torch.long),
        }


def collate_fn(batch):
    lens = torch.tensor([len(b["input"]) for b in batch])
    maxlen = lens.max()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lens[i]] = b["input"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "label": labels}


# --------------------------------------------------------------------- #
# load data
root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
root = root if root.exists() else pathlib.Path("SPR_BENCH/")
data = load_spr_bench(root)
vocab = build_vocab(data["train"])
itos = {i: s for s, i in vocab.items()}
num_classes = len(set(data["train"]["label"]))
vocab_size = len(vocab)
print("Dataset sizes:", {k: len(v) for k, v in data.items()})

bs = 128
train_dl = DataLoader(SPRSet(data["train"], vocab), bs, True, collate_fn=collate_fn)
dev_dl = DataLoader(SPRSet(data["dev"], vocab), bs, False, collate_fn=collate_fn)
test_dl = DataLoader(SPRSet(data["test"], vocab), bs, False, collate_fn=collate_fn)


# --------------------------------------------------------------------- #
# model
class MultiAttnRuleNet(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128, n_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.attn_q = nn.Parameter(torch.randn(n_cls, hid * 2))
        self.cls_vec = nn.Parameter(torch.randn(n_cls, hid * 2))
        self.bias = nn.Parameter(torch.zeros(n_cls))

    def forward(self, x, lens, need_attn=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,S,H)
        B, S, H = out.size()
        scores = torch.tanh(out) @ self.attn_q.T  # (B,S,C)
        mask = (x == 0).unsqueeze(-1)  # (B,S,1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)  # (B,S,C)
        context = torch.einsum("bsh,bsc->bch", out, weights)  # (B,C,H)
        logits = (context * self.cls_vec).sum(-1) + self.bias  # (B,C)
        if need_attn:
            return logits, weights  # weights (B,S,C)
        return logits


# --------------------------------------------------------------------- #
def run_epoch(model, dl, optim=None, ent_coeff=0.01):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits, attn = model(batch["input"], batch["lengths"], need_attn=True)
        ce = nn.functional.cross_entropy(logits, batch["label"])
        ent = (-attn * torch.log(attn + 1e-9)).sum() / attn.size(0)
        loss = ce + ent_coeff * ent if train else ce
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += ce.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    return tot_loss / len(dl.dataset), f1_score(gts, preds, average="macro"), preds, gts


# --------------------------------------------------------------------- #
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": [], "REA_dev": None, "REA_test": None},
        "rules": {},
        "predictions": [],
        "ground_truth": [],
    }
}

model = MultiAttnRuleNet(vocab_size, 64, 128, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 12
best_f1 = 0.0
best_state = None

for epoch in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, dev_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict()
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  val_f1 = {val_f1:.4f}")

model.load_state_dict(best_state)


# --------------------------------------------------------------------- #
# rule extraction (top-char per class)
def extract_rules(model, dl):
    model.eval()
    tok_scores = {c: np.zeros(vocab_size) for c in range(num_classes)}
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            _, attn = model(batch["input"], batch["lengths"], need_attn=True)  # (B,S,C)
            for seq, w in zip(batch["input"].cpu(), attn.cpu()):
                for pos, tok_id in enumerate(seq.tolist()):
                    if tok_id == 0:
                        break
                    for c in range(num_classes):
                        tok_scores[c][tok_id] += w[pos, c]
    rules = {c: int(np.argmax(tok_scores[c][1:]) + 1) for c in range(num_classes)}
    return rules


rules = extract_rules(model, train_dl)
experiment_data["SPR_BENCH"]["rules"] = {c: itos[t] for c, t in rules.items()}
print("Extracted character rules:", experiment_data["SPR_BENCH"]["rules"])


# --------------------------------------------------------------------- #
# rule-based classifier with neural fallback
def rule_predict(seq, rules):
    for ch in seq:
        for cls, tok in rules.items():
            if vocab.get(ch, 0) == tok:
                return cls
    return None


def evaluate_rules(split, rules, fallback_model=None):
    correct = 0
    total = len(split)
    preds = []
    fallback_model.eval()
    with torch.no_grad():
        for seq, label in zip(split["sequence"], split["label"]):
            p = rule_predict(seq, rules)
            if p is None:
                x = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(device)
                l = torch.tensor([len(seq)]).to(device)
                logit = fallback_model(x, l)
                p = int(torch.argmax(logit, 1).item())
            preds.append(p)
            correct += p == label
    return correct / total, preds


REA_dev, _ = evaluate_rules(data["dev"], rules, model)
REA_test, pt = evaluate_rules(data["test"], rules, model)
experiment_data["SPR_BENCH"]["metrics"]["REA_dev"] = REA_dev
experiment_data["SPR_BENCH"]["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Accuracy (dev):  {REA_dev:.4f}")
print(f"Rule Extraction Accuracy (test): {REA_test:.4f}")

# --------------------------------------------------------------------- #
# final neural model test F1
test_loss, test_f1, preds, gts = run_epoch(model, test_dl)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts
print(f"Neural Model Test Macro-F1: {test_f1:.4f}")

# --------------------------------------------------------------------- #
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
