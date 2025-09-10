import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
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


spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_path = spr_path if spr_path.exists() else pathlib.Path("SPR_BENCH/")
data = load_spr_bench(spr_path)
print({k: len(v) for k, v in data.items()})


# ---------------------------------------------------------------------
def build_vocab(split):
    charset = set()
    for seq in split["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(data["train"])
itos = {i: s for s, i in vocab.items()}
num_classes = len(set(data["train"]["label"]))
vocab_size = len(vocab)


# ---------------------------------------------------------------------
class SPRSet(Dataset):
    def __init__(self, ds, vocab):
        self.seqs, self.labels, self.vocab = ds["sequence"], ds["label"], vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = [self.vocab[c] for c in self.seqs[idx]]
        return {
            "input": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    lens = torch.tensor([len(b["input"]) for b in batch])
    maxlen = lens.max()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lens[i]] = b["input"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "label": labels}


bs = 128
train_dl = DataLoader(
    SPRSet(data["train"], vocab), bs, shuffle=True, collate_fn=collate_fn
)
dev_dl = DataLoader(
    SPRSet(data["dev"], vocab), bs, shuffle=False, collate_fn=collate_fn
)
test_dl = DataLoader(
    SPRSet(data["test"], vocab), bs, shuffle=False, collate_fn=collate_fn
)


# ---------------------------------------------------------------------
class AttnBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128, n_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(hid * 2))
        self.fc = nn.Linear(hid * 2, n_cls)

    def forward(self, x, lens, need_attn=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        attn_scores = torch.tanh(out) @ self.attn_vec
        mask = x != 0
        attn_scores[~mask] = -1e9
        weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(1)
        logits = self.fc(context)
        if need_attn:
            return logits, weights.squeeze(-1)
        return logits


# ---------------------------------------------------------------------
def run_epoch(model, dl, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"])
        loss = nn.functional.cross_entropy(logits, batch["label"])
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    return tot_loss / len(dl.dataset), f1_score(gts, preds, average="macro"), preds, gts


# ---------------------------------------------------------------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": [], "REA_dev": None, "REA_test": None},
        "rules": {},
        "preds_test": [],
        "gts_test": [],
    }
}
# ---------------------------------------------------------------------
model = AttnBiLSTM(vocab_size, 64, 128, num_classes).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 15
best_dev_f1 = 0.0
best_state = None
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optim)
    val_loss, val_f1, _, _ = run_epoch(model, dev_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_dev_f1:
        best_dev_f1 = val_f1
        best_state = model.state_dict()
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f}  val_f1={val_f1:.4f}")
# ---------------------------------------------------------------------
model.load_state_dict(best_state)


# ---------- RULE MINING ----------
def extract_rules(model, dl):
    model.eval()
    # sum attn per token per class
    token_scores = {c: np.zeros(vocab_size) for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}
    with torch.no_grad():
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, attn = model(batch["input"], batch["lengths"], need_attn=True)
            for seq, aw, label in zip(
                batch["input"].cpu(), attn.cpu(), batch["label"].cpu()
            ):
                counts[label.item()] += 1
                for tok, score in zip(seq.tolist(), aw.tolist()):
                    token_scores[label.item()][tok] += score
    rules = {}
    for c in range(num_classes):
        # ignore PAD(0)
        best_tok = int(np.argmax(token_scores[c][1:]) + 1)
        rules[c] = best_tok
    return rules


rules = extract_rules(model, train_dl)
experiment_data["SPR_BENCH"]["rules"] = {c: itos[tok] for c, tok in rules.items()}
print("Extracted rules (class -> char):", experiment_data["SPR_BENCH"]["rules"])


# ---------- RULE EVAL ----------
def rule_predict(seq, rules):
    present = {c: False for c in rules}
    for ch in seq:
        for cls, tok in rules.items():
            if vocab.get(ch, 0) == tok:
                present[cls] = True
    for cls, flag in present.items():
        if flag:
            return cls
    return None


def evaluate_rules(split, rules, fallback_model=None):
    correct = 0
    total = len(split)
    if fallback_model:
        fallback_model.eval()
    preds = []
    with torch.no_grad():
        for seq, label in zip(split["sequence"], split["label"]):
            rp = rule_predict(seq, rules)
            if rp is None and fallback_model is not None:
                # fallback
                enc = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(
                    device
                )
                lens = torch.tensor([len(seq)]).to(device)
                logit = fallback_model(enc, lens)
                rp = int(torch.argmax(logit, 1).item())
            preds.append(rp)
            if rp == label:
                correct += 1
    return correct / total, preds


REA_dev, _ = evaluate_rules(data["dev"], rules, model)
REA_test, preds_test = evaluate_rules(data["test"], rules, model)
experiment_data["SPR_BENCH"]["metrics"]["REA_dev"] = REA_dev
experiment_data["SPR_BENCH"]["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Accuracy (dev) : {REA_dev:.4f}")
print(f"Rule Extraction Accuracy (test): {REA_test:.4f}")
# ---------- Test F1 ----------
test_loss, test_f1, preds, gts = run_epoch(model, test_dl)
experiment_data["SPR_BENCH"]["preds_test"] = preds
experiment_data["SPR_BENCH"]["gts_test"] = gts
print(f"Neural Model Test Macro-F1: {test_f1:.4f}")
# ---------- Save metrics ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
# ---------- Simple plot ----------
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["metrics"]["val_f1"], label="Val_MacroF1")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Validation Macro-F1")
plt.legend()
plt.savefig(os.path.join(working_dir, "val_f1_curve.png"))
plt.close()
