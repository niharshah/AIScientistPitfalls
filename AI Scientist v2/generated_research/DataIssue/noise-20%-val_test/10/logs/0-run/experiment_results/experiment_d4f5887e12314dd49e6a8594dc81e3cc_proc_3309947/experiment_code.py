import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ---------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- reproducibility ----------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------- dataset loading ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
if not spr_root.exists():
    spr_root = pathlib.Path("SPR_BENCH/")
dsets = load_spr_bench(spr_root)


# ---------- vocab ----------
def build_vocab(hf_split):
    charset = set()
    for seq in hf_split["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(dsets["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(dsets["train"]["label"]))


# ---------- Dataset wrappers ----------
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
    maxlen = lens.max()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    bag = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    for i, b in enumerate(batch):
        l = b["seq"].size(0)
        padded[i, :l] = b["seq"]
        bag[i].index_add_(0, b["seq"], torch.ones(l))
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "bag": bag, "label": labels}


bs = 128
train_dl = DataLoader(
    SPRDataset(dsets["train"], vocab), bs, shuffle=True, collate_fn=collate
)
dev_dl = DataLoader(
    SPRDataset(dsets["dev"], vocab), bs, shuffle=False, collate_fn=collate
)
test_dl = DataLoader(
    SPRDataset(dsets["test"], vocab), bs, shuffle=False, collate_fn=collate
)


# ---------- model ----------
class AttnUniLSTM(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(hid))
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, x, lens, need_att=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        attn_scores = torch.tanh(out) @ self.attn_vec  # (B,L)
        mask = x != 0
        attn_scores[~mask] = -1e9
        w = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        ctx = (out * w).sum(1)
        logits = self.fc(ctx)
        return (logits, w.squeeze(-1)) if need_att else logits


class BagClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.lin = nn.Linear(vocab_sz, num_classes)

    def forward(self, bag):
        return self.lin(bag)


class HybridModel(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.seq_module = AttnUniLSTM(vocab_sz)
        self.bag_module = BagClassifier(vocab_sz)

    def forward(self, inp, lens, bag):
        seq_logits = self.seq_module(inp, lens)
        bag_logits = self.bag_module(bag)
        return (seq_logits + bag_logits) / 2.0, bag_logits


model = HybridModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l1_lambda = 1e-4


# ---------- helpers ----------
def epoch_pass(model, dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in dl:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits, bag_logits = model(batch["input"], batch["lengths"], batch["bag"])
        loss = nn.functional.cross_entropy(logits, batch["label"])
        loss += l1_lambda * model.bag_module.lin.weight.abs().mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    return total_loss / len(dl.dataset), macro_f1, preds, gts


# ---------- rule utilities ----------
def extract_rules(model):
    with torch.no_grad():
        w = model.bag_module.lin.weight.detach().cpu().numpy()  # (C,V)
    rules = {c: int(np.argmax(w[c, 1:]) + 1) for c in range(num_classes)}
    return rules  # token id per class


def rule_predict(seq, rules):
    for ch in seq:
        vid = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if vid == tok:
                return cls
    return None


def interpretable_accuracy(hf_split, rules, preds):
    """percentage where model correct and rule returns same label"""
    correct = 0
    for seq, true, model_pred in zip(hf_split["sequence"], hf_split["label"], preds):
        rule_out = rule_predict(seq, rules)
        if model_pred == true and rule_out == model_pred:
            correct += 1
    return correct / len(hf_split)


# ---------- experiment storage ----------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {
            "train_f1": [],
            "val_f1": [],
            "Interpretable_Acc": [],
        },
        "predictions": [],
        "ground_truth": [],
        "rules": {},
    }
}

# ---------- training loop ----------
best_f1 = 0.0
best_state = None
num_epochs = 12
for epoch in range(1, num_epochs + 1):
    tr_loss, tr_f1, _, _ = epoch_pass(model, train_dl, train=True)
    val_loss, val_f1, val_preds, _ = epoch_pass(model, dev_dl, train=False)

    # rules and interpretable metric
    rules = extract_rules(model)
    ia = interpretable_accuracy(dsets["dev"], rules, val_preds)

    # logging
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["metrics"]["Interpretable_Acc"].append(ia)

    print(
        f"Epoch {epoch:02d}: val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  IA={ia:.4f}"
    )

    # --------- BUGFIX: correct handling of best_f1 ----------
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# restore best model
model.load_state_dict(best_state)

# ---------- final evaluation on test ----------
test_loss, test_f1, test_preds, test_gts = epoch_pass(model, test_dl, train=False)
rules_final = extract_rules(model)
ia_test = interpretable_accuracy(dsets["test"], rules_final, test_preds)

print(
    f"Test: loss={test_loss:.4f}  Macro-F1={test_f1:.4f}  Interpretable-Acc={ia_test:.4f}"
)

# ---------- store experiment results ----------
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts
experiment_data["SPR_BENCH"]["rules"] = {c: itos[tok] for c, tok in rules_final.items()}

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
