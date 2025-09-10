import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------------------------------------------------------#
#  basic setup & reproducibility
# ------------------------------------------------------------------#
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------------------------------------------------------#
#  dataset loading helpers
# ------------------------------------------------------------------#
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
spr_root = spr_root if spr_root.exists() else pathlib.Path("SPR_BENCH/")
dsets = load_spr_bench(spr_root)
print({k: len(v) for k, v in dsets.items()})


# ------------------------------------------------------------------#
#  vocab utilities
# ------------------------------------------------------------------#
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


# ------------------------------------------------------------------#
#  pytorch dataset
# ------------------------------------------------------------------#
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
    lengths = torch.tensor([len(b["seq"]) for b in batch])
    max_len = lengths.max()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    bags = torch.zeros(len(batch), vocab_size, dtype=torch.float32)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        l = lengths[i]
        padded[i, :l] = b["seq"]
        bags[i].index_add_(0, b["seq"], torch.ones(l))
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


# ------------------------------------------------------------------#
#  model definitions
# ------------------------------------------------------------------#
class AttnBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid, bidirectional=True, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(hid * 2))
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x, lengths, need_attn=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        attn = torch.tanh(out) @ self.attn_vec  # (B,T)
        mask = x != 0
        attn[~mask] = -1e9
        w = torch.softmax(attn, 1).unsqueeze(-1)
        ctx = (out * w).sum(1)
        logits = self.fc(ctx)
        return (logits, w.squeeze(-1)) if need_attn else logits


class BagClassifier(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.lin = nn.Linear(vocab_sz, num_classes)

    def forward(self, bag):
        return self.lin(bag)


class HybridModel(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.bilstm = AttnBiLSTM(vocab_sz)
        self.bag = BagClassifier(vocab_sz)

    def forward(self, inp, lengths, bag):
        log_lstm = self.bilstm(inp, lengths)
        log_bag = self.bag(bag)
        return (log_lstm + log_bag) / 2, log_bag


# ------------------------------------------------------------------#
#  instantiate model & freeze embeddings (ablation)
# ------------------------------------------------------------------#
model = HybridModel(vocab_size).to(device)
# -------- Freeze character embeddings --------
model.bilstm.emb.weight.requires_grad = False

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
l1_lambda = 1e-4


# ------------------------------------------------------------------#
#  helpers for train / eval
# ------------------------------------------------------------------#
def epoch_pass(model, dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits, logits_bag = model(batch["input"], batch["lengths"], batch["bag"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
            loss += l1_lambda * model.bag.lin.weight.abs().mean()  # sparsity penalty
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return (
        total_loss / len(dl.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ------------------------------------------------------------------#
#  experiment tracking dict
# ------------------------------------------------------------------#
experiment_data = {
    "freeze_char_emb": {
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
exp_rec = experiment_data["freeze_char_emb"]["SPR_BENCH"]

# ------------------------------------------------------------------#
#  training loop
# ------------------------------------------------------------------#
best_f1, best_state = 0.0, None
epochs = 12
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = epoch_pass(model, train_dl, train=True)
    val_loss, val_f1, _, _ = epoch_pass(model, dev_dl, train=False)
    exp_rec["losses"]["train"].append(tr_loss)
    exp_rec["losses"]["val"].append(val_loss)
    exp_rec["metrics"]["train_f1"].append(tr_f1)
    exp_rec["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    print(f"Epoch {ep:02d} | val_loss={val_loss:.4f} | val_f1={val_f1:.4f}")

model.load_state_dict(best_state, strict=False)

# ------------------------------------------------------------------#
#  rule extraction from bag-of-chars head
# ------------------------------------------------------------------#
with torch.no_grad():
    weights = model.bag.lin.weight.detach().cpu().numpy()  # (C,V)
rules = {c: int(np.argmax(weights[c, 1:]) + 1) for c in range(num_classes)}  # skip PAD
exp_rec["rules"] = {c: itos[tok] for c, tok in rules.items()}
print("Learned single-token rules:", exp_rec["rules"])


def rule_predict(seq, rules_d):
    for ch in seq:
        vid = vocab.get(ch, 0)
        for cls, tok in rules_d.items():
            if vid == tok:
                return cls
    return None


def evaluate_rules(hf_split, rules_d, model):
    correct, total = 0, len(hf_split)
    preds = []
    model.eval()
    with torch.no_grad():
        for seq, label in zip(hf_split["sequence"], hf_split["label"]):
            rp = rule_predict(seq, rules_d)
            if rp is None:
                ids = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(
                    device
                )
                lens = torch.tensor([ids.size(1)]).to(device)
                bag = torch.zeros(1, vocab_size, device=device)
                bag[0].index_add_(
                    0, ids.squeeze(0), torch.ones(ids.size(1), device=device)
                )
                logit, _ = model(ids, lens, bag)
                rp = int(torch.argmax(logit, 1).item())
            preds.append(rp)
            if rp == label:
                correct += 1
    return correct / total, preds


REA_dev, _ = evaluate_rules(dsets["dev"], rules, model)
REA_test, ptest = evaluate_rules(dsets["test"], rules, model)
exp_rec["metrics"]["REA_dev"] = REA_dev
exp_rec["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Acc (dev):  {REA_dev:.4f}")
print(f"Rule Extraction Acc (test): {REA_test:.4f}")

# ------------------------------------------------------------------#
#  final test evaluation
# ------------------------------------------------------------------#
test_loss, test_f1, preds, gts = epoch_pass(model, test_dl, False)
exp_rec["preds_test"] = preds
exp_rec["gts_test"] = gts
print(f"Hybrid Model Test Macro-F1: {test_f1:.4f}")

# ------------------------------------------------------------------#
#  save experiment data
# ------------------------------------------------------------------#
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
