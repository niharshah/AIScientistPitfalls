import os, pathlib, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score

# ------------------ experiment dict ---------------------------------
experiment_data = {
    "Remove_BoC_Branch": {
        "SPR_BENCH": {
            "losses": {"train": [], "val": []},
            "metrics": {
                "train_f1": [],
                "val_f1": [],
                "REA_dev": None,
                "REA_test": None,
            },
            "rules": {},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ------------------ misc setup --------------------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ------------------ SPR-BENCH loading -------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_name):
        return load_dataset(
            "csv",
            data_files=str(root / f"{split_name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    for s in ["train", "dev", "test"]:
        d[s] = _load(s)
    return d


spr_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr_root = spr_root if spr_root.exists() else pathlib.Path("SPR_BENCH/")
dsets = load_spr_bench(spr_root)
print({k: len(v) for k, v in dsets.items()})


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


# ------------------ dataset / dataloader ----------------------------
class SPRDataset(Dataset):
    def __init__(self, hf_ds, vocab):
        self.seqs = hf_ds["sequence"]
        self.labels = hf_ds["label"]
        self.vocab = vocab

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor([self.vocab[c] for c in self.seqs[idx]], dtype=torch.long)
        return {"seq": seq, "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def collate(batch):
    lens = torch.tensor([len(b["seq"]) for b in batch])
    maxlen = lens.max()
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lens[i]] = b["seq"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input": padded, "lengths": lens, "label": labels}


bs = 128
train_dl = DataLoader(SPRDataset(dsets["train"], vocab), bs, True, collate_fn=collate)
dev_dl = DataLoader(SPRDataset(dsets["dev"], vocab), bs, False, collate_fn=collate)
test_dl = DataLoader(SPRDataset(dsets["test"], vocab), bs, False, collate_fn=collate)


# ------------------ BiLSTM-attention model --------------------------
class AttnBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.att_vec = nn.Parameter(torch.randn(hid * 2))
        self.fc = nn.Linear(hid * 2, num_classes)

    def forward(self, x, lens, need_attn=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        attn = torch.tanh(out) @ self.att_vec
        mask = x != 0
        attn[~mask] = -1e9
        w = torch.softmax(attn, 1).unsqueeze(-1)  # (B,L,1)
        ctx = (out * w).sum(1)  # (B,2H)
        logits = self.fc(ctx)
        return (logits, w.squeeze(-1)) if need_attn else logits


model = AttnBiLSTM(vocab_size).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 12
best_f1, best_state = 0.0, None


# ------------------ train / eval loop -------------------------------
def epoch_pass(model, dl, train=False):
    model.train() if train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input"], batch["lengths"])
            loss = nn.functional.cross_entropy(logits, batch["label"])
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            total_loss += loss.item() * batch["label"].size(0)
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return total_loss / len(dl.dataset), f1_score(gts, preds, average="macro")


for ep in range(1, epochs + 1):
    tr_loss, tr_f1 = epoch_pass(model, train_dl, True)
    val_loss, val_f1 = epoch_pass(model, dev_dl, False)
    ed = experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]
    ed["losses"]["train"].append(tr_loss)
    ed["losses"]["val"].append(val_loss)
    ed["metrics"]["train_f1"].append(tr_f1)
    ed["metrics"]["val_f1"].append(val_f1)
    if val_f1 > best_f1:
        best_f1, best_state = val_f1, model.state_dict()
    print(f"Epoch {ep}: val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

model.load_state_dict(best_state)

# ------------------ rule extraction via attention -------------------
token_scores = np.zeros((num_classes, vocab_size), dtype=np.float32)
train_dl_attn = DataLoader(
    SPRDataset(dsets["train"], vocab), bs, False, collate_fn=collate
)
model.eval()
with torch.no_grad():
    for batch in train_dl_attn:
        lens = batch["lengths"].to(device)
        inp = batch["input"].to(device)
        labels = batch["label"].numpy()
        _, att = model(inp, lens, need_attn=True)
        att = att.cpu().numpy()
        for i in range(inp.size(0)):
            seq = inp[i].cpu().numpy()
            for pos, vid in enumerate(seq[: lens[i]]):
                token_scores[labels[i], vid] += att[i, pos]

rules = {}
for c in range(num_classes):
    tok = int(np.argmax(token_scores[c, 1:]) + 1) if token_scores[c, 1:].sum() else 0
    rules[c] = tok
experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["rules"] = {
    c: itos.get(t, "<PAD>") for c, t in rules.items()
}
print("Extracted rules:", experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["rules"])


# ------------------ rule evaluation helper --------------------------
def rule_predict(seq, rules):
    for ch in seq:
        vid = vocab.get(ch, 0)
        for cls, tok in rules.items():
            if vid == tok and tok != 0:
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
                inp = torch.tensor([[vocab[c] for c in seq]], dtype=torch.long).to(
                    device
                )
                lens = torch.tensor([inp.size(1)], device=device)
                logit = model(inp, lens)
                rp = int(torch.argmax(logit, 1).item())
            preds.append(rp)
            correct += rp == label
    return correct / total, preds


REA_dev, _ = evaluate_rules(dsets["dev"], rules, model)
REA_test, preds_test = evaluate_rules(dsets["test"], rules, model)
experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["metrics"]["REA_dev"] = REA_dev
experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Accuracy (dev): {REA_dev:.4f}")
print(f"Rule Extraction Accuracy (test): {REA_test:.4f}")

# ------------------ final test metrics ------------------------------
test_loss, test_f1 = epoch_pass(model, test_dl, False)
print(f"BiLSTM-only Test Macro-F1: {test_f1:.4f}")
experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["predictions"] = preds_test
experiment_data["Remove_BoC_Branch"]["SPR_BENCH"]["ground_truth"] = dsets["test"][
    "label"
]

# ------------------ save --------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
