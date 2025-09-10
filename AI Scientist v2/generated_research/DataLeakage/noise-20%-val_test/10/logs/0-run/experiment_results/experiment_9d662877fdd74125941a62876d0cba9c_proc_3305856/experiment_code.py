import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------------------------------------------------ #
# workspace & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------ #
# 1. data loading
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


# ------------------------------------------------------------ #
# 2. vocabulary
def build_vocab(ds):
    charset = set()
    for seq in ds["sequence"]:
        charset.update(seq)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(dsets["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(dsets["train"]["label"]))


# ------------------------------------------------------------ #
# 3. Torch dataset
class SPRDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.seq = hf_ds["sequence"]
        self.label = hf_ds["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = [self.vocab[c] for c in self.seq[idx]]
        return {
            "seq": torch.tensor(s, dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.long),
        }


def collate(batch):
    lengths = torch.tensor([len(x["seq"]) for x in batch])
    maxlen = lengths.max()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    bags = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        seq = b["seq"]
        padded[i, : len(seq)] = seq
        bags[i].index_add_(0, seq, torch.ones_like(seq, dtype=torch.float32))
    return {"input": padded, "lengths": lengths, "bag": bags, "label": labels}


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


# ------------------------------------------------------------ #
# 4. model
class HybridRuleNet(nn.Module):
    def __init__(self, vocab_sz, num_cls, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.att_vec = nn.Parameter(torch.randn(hid * 2))
        self.classifier_neural = nn.Linear(hid * 2, num_cls)
        self.rule_head = nn.Linear(vocab_sz, num_cls, bias=False)  # bag-of-char

    def forward(self, x, lengths, bag, need_attn=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        att = torch.tanh(out) @ self.att_vec  # [B,L]
        mask = x != 0
        att[~mask] = -1e9
        w = torch.softmax(att, dim=1).unsqueeze(-1)
        ctx = (out * w).sum(1)
        neural_logits = self.classifier_neural(ctx)
        rule_logits = self.rule_head(bag)
        logits = neural_logits + rule_logits
        return (logits, w.squeeze(-1)) if need_attn else logits


model = HybridRuleNet(vocab_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------ #
# 5. experiment tracking
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": [], "REA_dev": []},
        "rules_epochs": [],
    }
}


# ------------------------------------------------------------ #
# 6. helper functions
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in dl:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model(batch["input"], batch["lengths"], batch["bag"])
        loss = nn.functional.cross_entropy(logits, batch["label"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    return tot_loss / len(dl.dataset), f1_score(gts, preds, average="macro")


def extract_rules():
    with torch.no_grad():
        W = model.rule_head.weight.data.cpu().numpy()  # [C, V]
    rules = {}
    for c in range(num_classes):
        char_idx = int(np.argmax(W[c, 1:]) + 1)  # skip PAD
        rules[c] = char_idx
    return rules


def rule_predict_seq(seq_chars, rules):
    # returns predicted class or None
    best_cls, best_weight = None, -1e9
    for ch in seq_chars:
        idx = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if idx == tok:
                weight = model.rule_head.weight[cls, tok].item()
                if weight > best_weight:
                    best_weight, best_cls = weight, cls
    return best_cls


def compute_REA(split, rules):
    correct = 0
    for seq, label in zip(split["sequence"], split["label"]):
        pred = rule_predict_seq(seq, rules)
        correct += int(pred == label)
    return correct / len(split)


# ------------------------------------------------------------ #
# 7. training loop
epochs = 15
best_val_f1, best_state = 0.0, None
for epoch in range(1, epochs + 1):
    tr_loss, tr_f1 = run_epoch(train_dl, train=True)
    val_loss, val_f1 = run_epoch(dev_dl, train=False)
    rules = extract_rules()
    rea_dev = compute_REA(dsets["dev"], rules)
    # log
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["metrics"]["REA_dev"].append(rea_dev)
    experiment_data["SPR_BENCH"]["rules_epochs"].append(
        {c: itos[t] for c, t in rules.items()}
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_f1={val_f1:.4f} | REA_dev={rea_dev:.3f}"
    )
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# ------------------------------------------------------------ #
# 8. evaluation on test
model.load_state_dict(best_state)
test_loss, test_f1 = run_epoch(test_dl, train=False)
final_rules = extract_rules()
rea_test = compute_REA(dsets["test"], final_rules)
print(f"\nBest neural model Test Macro-F1: {test_f1:.4f}")
print(f"Rule Extraction Accuracy (test): {rea_test:.4f}")
print(
    "Learned character rules per class:", {c: itos[t] for c, t in final_rules.items()}
)

# ------------------------------------------------------------ #
# 9. save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
