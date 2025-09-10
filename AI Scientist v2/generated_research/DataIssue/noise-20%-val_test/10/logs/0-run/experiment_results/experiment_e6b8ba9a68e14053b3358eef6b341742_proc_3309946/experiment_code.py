import os, pathlib, random, copy, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ----------------------------- housekeeping -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------- data loading -----------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
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


# ----------------------------- vocabulary -----------------------------
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


# ----------------------------- dataset objects -----------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.seqs = hf_split["sequence"]
        self.labels = hf_split["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = [self.vocab[c] for c in self.seqs[idx]]
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch):
    lens = torch.tensor([len(b["seq"]) for b in batch], dtype=torch.long)
    max_len = lens.max()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
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


# ----------------------------- model definitions -----------------------------
class AttnUniLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(hid_dim))
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,L,H)
        attn_scores = torch.tanh(out) @ self.attn_vec  # (B,L)
        mask = x != 0
        attn_scores[~mask] = -1e9
        w = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B,L,1)
        ctx = (out * w).sum(1)  # (B,H)
        return self.fc(ctx)


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

    def forward(self, inp, lengths, bag):
        log_seq = self.seq_model(inp, lengths)
        log_bag = self.bag_model(bag)
        return (log_seq + log_bag) / 2.0, log_bag  # ensemble & standalone-bag


model = HybridModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l1_lambda = 1e-4


# ----------------------------- helpers -----------------------------
def move_batch_to_device(batch):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def pass_epoch(dataloader, train=False):
    model.train() if train else model.eval()
    epoch_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in dataloader:
            batch = move_batch_to_device(batch)
            logits, logits_bag = model(batch["input"], batch["lengths"], batch["bag"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
            loss += l1_lambda * model.bag_model.lin.weight.abs().mean()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * batch["label"].size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    avg_loss = epoch_loss / len(dataloader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


def extract_rules(model):
    # pick the highest-weight token (excluding PAD) per class from bag classifier
    with torch.no_grad():
        w = model.bag_model.lin.weight.detach().cpu().numpy()
    rules = {}
    for cls in range(num_classes):
        best_tok = int(np.argmax(w[cls, 1:]) + 1)  # +1 to skip PAD
        rules[cls] = best_tok
    return rules


def rule_only_predict(seq, rules):
    for ch in seq:
        tok_id = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if tok_id == tok:
                return cls
    return None


def interpretable_accuracy(hf_split, rules, model):
    model.eval()
    correct_and_faithful = 0
    total = len(hf_split)
    with torch.no_grad():
        for seq, true_label in zip(hf_split["sequence"], hf_split["label"]):
            # model prediction
            inp = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(device)
            length = torch.tensor([inp.size(1)], dtype=torch.long).to(device)
            bag = torch.zeros(1, vocab_size, device=device)
            bag[0].index_add_(0, inp.squeeze(0), torch.ones(inp.size(1), device=device))
            logit, _ = model(inp, length, bag)
            model_pred = int(torch.argmax(logit, dim=1))
            # rule prediction
            rule_pred = rule_only_predict(seq, rules)
            if rule_pred is None:
                continue
            if (model_pred == rule_pred) and (model_pred == true_label):
                correct_and_faithful += 1
    return correct_and_faithful / total


# ----------------------------- experiment data container -----------------------------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {
            "train_f1": [],
            "val_f1": [],
            "val_interpretable_acc": [],
            "test_interpretable_acc": None,
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# ----------------------------- training loop -----------------------------
best_f1 = 0.0
best_state = None
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_f1, _, _ = pass_epoch(train_dl, train=True)
    val_loss, val_f1, _, _ = pass_epoch(dev_dl, train=False)

    # rule extraction and interpretable accuracy on dev
    current_rules = extract_rules(model)
    val_IA = interpretable_accuracy(dsets["dev"], current_rules, model)

    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_interpretable_acc"].append(val_IA)

    if val_f1 > best_f1:
        best_f1 = val_f1  # bugfix: simple assignment, not unpacking
        best_state = copy.deepcopy(model.state_dict())

    dt = time.time() - t0
    print(
        f"Epoch {epoch:02d} | {dt:5.1f}s | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_f1={val_f1:.4f} IA={val_IA:.4f}"
    )

# ----------------------------- evaluation on test -----------------------------
model.load_state_dict(best_state)

test_loss, test_f1, test_preds, test_gts = pass_epoch(test_dl, train=False)
experiment_data["SPR_BENCH"]["predictions"] = test_preds
experiment_data["SPR_BENCH"]["ground_truth"] = test_gts

# Interpretable-Accuracy on test
best_rules = extract_rules(model)
test_IA = interpretable_accuracy(dsets["test"], best_rules, model)
experiment_data["SPR_BENCH"]["metrics"]["test_interpretable_acc"] = test_IA

print(
    f"Best-model Test results: loss={test_loss:.4f}  F1={test_f1:.4f}  IA={test_IA:.4f}"
)
print(
    "Extracted rules (token → char):",
    {cls: itos[tok] for cls, tok in best_rules.items()},
)

# ----------------------------- save everything -----------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
