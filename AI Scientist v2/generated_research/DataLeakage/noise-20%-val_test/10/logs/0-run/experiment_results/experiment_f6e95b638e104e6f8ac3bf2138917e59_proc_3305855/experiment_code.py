import os, pathlib, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------
# ---------------  DATA -------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    d = DatasetDict()
    for split in ["train", "dev", "test"]:
        d[split] = _load(f"{split}.csv")
    return d


# try a few default locations
candidate = [
    pathlib.Path("SPR_BENCH"),
    pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH"),
]
for c in candidate:
    if (c / "train.csv").exists():
        data_root = c
        break
else:
    raise FileNotFoundError("Cannot locate SPR_BENCH data folder")
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})


# ---------------- VOCAB ------------------------------------
def build_vocab(split):
    charset = set()
    for s in split["sequence"]:
        charset.update(s)
    stoi = {c: i + 1 for i, c in enumerate(sorted(charset))}
    stoi["<PAD>"] = 0
    return stoi


vocab = build_vocab(spr["train"])
itos = {i: s for s, i in vocab.items()}
vocab_size = len(vocab)
num_classes = len(set(spr["train"]["label"]))


# --------------- DATASET -----------------------------------
class SPRSet(Dataset):
    def __init__(self, ds, vocab):
        self.data = ds
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]["sequence"]
        lab = self.data[idx]["label"]
        ids = [self.vocab[c] for c in seq]
        return {
            "input": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(lab, dtype=torch.long),
        }


def collate(batch):
    lens = [len(b["input"]) for b in batch]
    maxlen = max(lens)
    padded = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, b in enumerate(batch):
        padded[i, : lens[i]] = b["input"]
    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor(lens)
    return {"input": padded, "lengths": lengths, "label": labels}


bs = 256
train_dl = DataLoader(SPRSet(spr["train"], vocab), bs, shuffle=True, collate_fn=collate)
dev_dl = DataLoader(SPRSet(spr["dev"], vocab), bs, shuffle=False, collate_fn=collate)
test_dl = DataLoader(SPRSet(spr["test"], vocab), bs, shuffle=False, collate_fn=collate)


# --------------- MODEL -------------------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, ks=(2, 3, 4, 5), ch=128, num_cls=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, ch, k) for k in ks])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(ch * len(ks), num_cls)
        self.ks = ks

    def forward(self, x, lens, return_feats=False):
        # x:[B,L]
        e = self.emb(x).transpose(1, 2)  # [B,E,L]
        feats = []
        for conv in self.convs:
            f = torch.relu(conv(e))  # [B,C,L-k+1]
            f = torch.max(f, dim=2).values  # global max pool [B,C]
            feats.append(f)
        h = torch.cat(feats, dim=1)  # [B,C*len(ks)]
        h = self.dropout(h)
        logits = self.fc(h)
        if return_feats:
            return logits, h
        return logits


model = CharCNN(vocab_size, 64, (2, 3, 4, 5), 128, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)


# ------------- TRAIN / EVAL --------------------------------
def run_epoch(model, dl, opt=None):
    train = opt is not None
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
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["label"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["label"].cpu().tolist())
    f1 = f1_score(gts, preds, average="macro")
    return tot_loss / len(dl.dataset), f1, preds, gts


# ------------- TRAIN LOOP ----------------------------------
experiment_data = {
    "SPR_BENCH": {
        "losses": {"train": [], "val": []},
        "metrics": {"train_f1": [], "val_f1": [], "REA_dev": None, "REA_test": None},
        "rules": {},
        "preds_test": [],
        "gts_test": [],
    }
}
best_f1, best_state = 0.0, None
patience, stale = 2, 0
epochs = 10
for ep in range(1, epochs + 1):
    tr_loss, tr_f1, _, _ = run_epoch(model, train_dl, optimizer)
    val_loss, val_f1, _, _ = run_epoch(model, dev_dl)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(tr_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)
    print(f"Epoch {ep}: validation_loss = {val_loss:.4f}  val_f1 = {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1, val_best = val_f1, ep
        best_state = model.state_dict()
        stale = 0
    else:
        stale += 1
    if stale >= patience:
        print("Early stopping.")
        break
model.load_state_dict(best_state)


# ------------- RULE EXTRACTION -----------------------------
def extract_rules(model, train_dl, top_per_class=5):
    model.eval()
    # store best n-gram per filter
    filter_best_act = [-1e9] * len(model.convs) * model.convs[0].out_channels
    filter_best_ngram = [""] * len(filter_best_act)
    filter_ks = []
    for k, ksize in enumerate(model.ks):
        filter_ks.extend([ksize] * model.convs[0].out_channels)
    with torch.no_grad():
        for batch in train_dl:
            seqs = batch["input"]
            lens = batch["lengths"]
            seqs_gpu = seqs.to(device)
            embed = model.emb(seqs_gpu).transpose(1, 2)
            offset = 0
            for conv, ksize in zip(model.convs, model.ks):
                conv_out = torch.relu(conv(embed))  # [B,C,L-k+1]
                conv_out_cpu = conv_out.cpu()
                for b in range(seqs.size(0)):
                    for f in range(conv_out_cpu.size(1)):
                        feat = conv_out_cpu[b, f]
                        val = torch.max(feat).item()
                        if val > filter_best_act[offset + f]:
                            j = torch.argmax(feat).item()
                            ids = seqs[b, j : j + ksize].tolist()
                            ngram = "".join([itos[i] for i in ids if i != 0])
                            filter_best_act[offset + f] = val
                            filter_best_ngram[offset + f] = ngram
                offset += conv_out_cpu.size(1)
    # now map filter -> class via fc weights
    fc_weight = model.fc.weight.data.cpu().numpy()  # [num_cls, F]
    rules_per_class = {c: [] for c in range(num_classes)}
    for f in range(fc_weight.shape[1]):
        cls = int(np.argmax(fc_weight[:, f]))
        rules_per_class[cls].append((fc_weight[cls, f], filter_best_ngram[f]))
    rules = {}
    for c in range(num_classes):
        sorted_rules = sorted(rules_per_class[c], key=lambda x: -abs(x[0]))
        for w, ng in sorted_rules[:top_per_class]:
            if ng != "":
                rules[ng] = c
    return rules


rules = extract_rules(model, train_dl, top_per_class=5)
experiment_data["SPR_BENCH"]["rules"] = rules
print("Extracted rules (string -> class):", rules)


# ------------- REA -----------------------------------------
def rule_only_predict(seq, rules):
    for ng, c in rules.items():
        if ng in seq:
            return c
    return None


def eval_rules(split, rules):
    correct = 0
    for seq, label in zip(split["sequence"], split["label"]):
        p = rule_only_predict(seq, rules)
        if p is not None and p == label:
            correct += 1
    return correct / len(split)


REA_dev = eval_rules(spr["dev"], rules)
REA_test = eval_rules(spr["test"], rules)
experiment_data["SPR_BENCH"]["metrics"]["REA_dev"] = REA_dev
experiment_data["SPR_BENCH"]["metrics"]["REA_test"] = REA_test
print(f"Rule Extraction Accuracy (dev) : {REA_dev:.4f}")
print(f"Rule Extraction Accuracy (test): {REA_test:.4f}")
# ------------- FINAL TEST METRIC ---------------------------
test_loss, test_f1, preds, gts = run_epoch(model, test_dl)
experiment_data["SPR_BENCH"]["preds_test"] = preds
experiment_data["SPR_BENCH"]["gts_test"] = gts
print(f"Neural Model Test Macro-F1: {test_f1:.4f}")
# ------------- SAVE METRICS & PLOT -------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
plt.figure()
plt.plot(experiment_data["SPR_BENCH"]["metrics"]["val_f1"], label="Val Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("Validation Performance")
plt.legend()
plt.savefig(os.path.join(working_dir, "val_f1_curve.png"))
plt.close()
