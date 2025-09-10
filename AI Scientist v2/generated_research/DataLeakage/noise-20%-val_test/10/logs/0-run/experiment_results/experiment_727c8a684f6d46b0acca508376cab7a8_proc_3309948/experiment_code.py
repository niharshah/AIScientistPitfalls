# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------------- misc & dirs ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility (optional, inexpensive)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ------------------------- load SPR-BENCH ---------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict({s: _load(f"{s}.csv") for s in ["train", "dev", "test"]})


spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_root = spr_root if spr_root.exists() else pathlib.Path("SPR_BENCH/")
dsets = load_spr_bench(spr_root)
print({k: len(v) for k, v in dsets.items()})


# ----------------------- vocab --------------------------------------
def build_vocab(split):
    charset = set()
    for seq in split["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(dsets["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(dsets["train"]["label"]))


# ----------------------- dataset & loader ---------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
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
    lens = torch.tensor([len(b["seq"]) for b in batch], dtype=torch.long)
    maxlen = lens.max().item()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    bags = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = lens[i]
        padded[i, :l] = b["seq"]
        bags[i].index_add_(0, b["seq"], torch.ones(l, dtype=torch.float32))
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "bag": bags, "label": labels}


bs = 128
train_dl = DataLoader(SPRDataset(dsets["train"], vocab), bs, True, collate_fn=collate)
dev_dl = DataLoader(SPRDataset(dsets["dev"], vocab), bs, False, collate_fn=collate)
test_dl = DataLoader(SPRDataset(dsets["test"], vocab), bs, False, collate_fn=collate)


# ----------------------- No-Attention BiLSTM ------------------------
class MeanBiLSTM(nn.Module):
    """
    BiLSTM followed by simple mean-pooling instead of learnable attention.
    """

    def __init__(self, vocab_sz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x, lens):
        e = self.emb(x)  # (B,T,E)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,T,2H)

        mask = (x != 0).unsqueeze(-1).float()  # (B,T,1)
        summed = (out * mask).sum(1)  # (B,2H)
        ctx = summed / lens.unsqueeze(-1).float()  # mean pooling
        logits = self.fc(ctx)  # (B,C)
        return logits  # no attention weights


# ----------------------- Bag-of-Characters branch -------------------
class BagClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.lin = nn.Linear(vocab_sz, num_classes)

    def forward(self, bag_vec):
        return self.lin(bag_vec)


# ----------------------- Hybrid Model -------------------------------
class HybridModel(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.bilstm = MeanBiLSTM(vocab_sz)
        self.bag = BagClassifier(vocab_sz)

    def forward(self, inp, lens, bag):
        log_lstm = self.bilstm(inp, lens)
        log_bag = self.bag(bag)
        return (log_lstm + log_bag) / 2, log_bag  # ensemble + raw BoC logits


model = HybridModel(vocab_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
l1_lambda = 1e-4


# ----------------------- helpers ------------------------------------
def epoch_pass(model, dl, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, logits_bag = model(batch["input"], batch["lengths"], batch["bag"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
            loss += l1_lambda * model.bag.lin.weight.abs().mean()  # sparsity on BoC
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            tot_loss += loss.item() * batch["label"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return tot_loss / len(dl.dataset), f1_score(gts, preds, average="macro"), preds, gts


# ----------------------- experiment tracking ------------------------
experiment_data = {
    "NoAttnBiLSTM": {
        "SPR_BENCH": {
            "losses": {"train": [], "val": []},
            "metrics": {
                "train_f1": [],
                "val_f1": [],
                "REA_dev": None,
                "REA_test": None,
            },
            "rules": {},
            "preds_test": [],
            "gts_test": [],
        }
    }
}

best_f1, best_state = 0.0, None
epochs = 12
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = epoch_pass(model, train_dl, True)
    val_loss, val_f1, _, _ = epoch_pass(model, dev_dl, False)
    ed = experiment_data["NoAttnBiLSTM"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_f1:
        best_f1, best_state = val_f1, model.state_dict()
    print(f"Epoch {ep:02d}: val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

model.load_state_dict(best_state)

# ----------------------- rule extraction (single token) -------------
with torch.no_grad():
    weights = model.bag.lin.weight.detach().cpu().numpy()  # (C,V)
rules = {c: int(np.argmax(weights[c, 1:]) + 1) for c in range(num_classes)}
experiment_data["NoAttnBiLSTM"]["SPR_BENCH"]["rules"] = {
    c: itos[t] for c, t in rules.items()
}
print(
    "Learned single-token rules:", experiment_data["NoAttnBiLSTM"]["SPR_BENCH"]["rules"]
)


def rule_predict(seq, rules):
    for ch in seq:
        vid = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if vid == tok:
                return cls
    return None


def evaluate_rules(hf_split, rules, model):
    correct, total = 0, len(hf_split)
    preds = []
    model.eval()
    with torch.no_grad():
        for seq, label in zip(hf_split["sequence"], hf_split["label"]):
            rp = rule_predict(seq, rules)
            if rp is None:
                # fallback to hybrid model
                ids = torch.tensor(
                    [[vocab[c] for c in seq]], dtype=torch.long, device=device
                )
                lens = torch.tensor([ids.size(1)], device=device)
                bag = torch.zeros(1, vocab_size, device=device)
                bag.index_add_(
                    1, ids.squeeze(0), torch.ones(ids.size(1), device=device)
                )
                logit, _ = model(ids, lens, bag)
                rp = int(torch.argmax(logit, 1).item())
            preds.append(rp)
            if rp == label:
                correct += 1
    return correct / total, preds


REA_dev, _ = evaluate_rules(dsets["dev"], rules, model)
REA_test, preds_te = evaluate_rules(dsets["test"], rules, model)
ed = experiment_data["NoAttnBiLSTM"]["SPR_BENCH"]
ed["metrics"]["REA_dev"] = REA_dev
ed["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Accuracy – dev:  {REA_dev:.4f}")
print(f"Rule Extraction Accuracy – test: {REA_test:.4f}")

# ----------------------- final evaluation on test -------------------
test_loss, test_f1, preds, gts = epoch_pass(model, test_dl, False)
ed["preds_test"] = preds
ed["gts_test"] = gts
print(f"Hybrid Model Test Macro-F1: {test_f1:.4f}")

# ----------------------- save everything ----------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
