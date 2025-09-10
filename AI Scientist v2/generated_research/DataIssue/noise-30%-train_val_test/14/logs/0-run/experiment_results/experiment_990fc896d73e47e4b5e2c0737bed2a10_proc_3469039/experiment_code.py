import os, pathlib, random, time, json, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, DatasetDict

# ------------------ working dir & device ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ reproducibility -----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ------------------ data loading --------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(f):
        return load_dataset(
            "csv", data_files=str(root / f), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# ------------------ vocab construction -------------------
CLS, PAD, UNK, MASK = "[CLS]", "[PAD]", "[UNK]", "[MASK]"
vocab = {PAD: 0, CLS: 1, UNK: 2, MASK: 3}


def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)


for seq in spr["train"]["sequence"]:
    for t in seq.split():
        add(t)
vocab_size = len(vocab)
labels = sorted(set(spr["train"]["label"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print(f"Vocab={vocab_size}, Labels={num_labels}")

# ------------------ encode utils -------------------------
MAX_LEN = 128


def encode(seq: str):
    toks = [CLS] + seq.split()
    ids = [vocab.get(t, vocab[UNK]) for t in toks][:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, hf):
        self.seq = hf["sequence"]
        self.lab = hf["label"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        ids, att = encode(self.seq[idx])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(att),
            "labels": torch.tensor(label2id[self.lab[idx]]),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


BATCH = 64
train_loader = DataLoader(
    SPRDataset(spr["train"]), BATCH, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), BATCH, shuffle=False, collate_fn=collate
)


# ------------------ model defs ---------------------------
class Encoder(nn.Module):
    def __init__(self, d_model=128, heads=4, layers=2, ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, ids, mask):
        x = self.emb(ids) + self.pos[:, : ids.size(1)]
        x = self.enc(x, src_key_padding_mask=~mask.bool())
        return x


class Baseline(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.encoder = Encoder(d_model)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, mask):
        h = self.encoder(ids, mask)
        return self.cls(h[:, 0])


class MultiTask(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.encoder = Encoder(d_model)
        self.cls_head = nn.Linear(d_model, num_labels)
        # MLM head ties weights with embedding
        self.lm_decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_decoder.weight = self.encoder.emb.weight

    def forward(self, ids, mask, mlm_labels=None):
        h = self.encoder(ids, mask)
        logits_cls = self.cls_head(h[:, 0])
        if mlm_labels is None:
            return logits_cls, None
        logits_mlm = self.lm_decoder(h)  # (B,L,V)
        return logits_cls, logits_mlm


# ------------------ masking util -------------------------
def create_mlm_inputs(input_ids, prob=0.15):
    ids = input_ids.clone()
    labels = torch.full_like(ids, -100)
    # skip CLS token at pos0
    rand = torch.rand_like(ids.float())
    mask_sel = (rand < prob) & (ids != vocab[PAD]) & (ids != vocab[CLS])
    labels[mask_sel] = ids[mask_sel]
    ids[mask_sel] = vocab[MASK]
    return ids, labels


# ------------------ training / evaluation ---------------
ce = nn.CrossEntropyLoss(ignore_index=-100)


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if isinstance(model, MultiTask):
            ids_mlm, mlm_lbl = create_mlm_inputs(batch["input_ids"])
            ids_mlm = ids_mlm.to(device)
            mlm_lbl = mlm_lbl.to(device)
            out_cls, out_mlm = model(ids_mlm, batch["attention_mask"], mlm_lbl)
            loss_cls = ce(out_cls, batch["labels"])
            loss_mlm = ce(out_mlm.view(-1, vocab_size), mlm_lbl.view(-1))
            loss = loss_cls + 0.5 * loss_mlm
        else:
            out_cls = model(batch["input_ids"], batch["attention_mask"])
            loss = ce(out_cls, batch["labels"])
            loss_cls, loss_mlm = loss, torch.tensor(0.0, device=device)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(out_cls, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    macro_f1 = f1_score(gts, preds, average="macro")
    acc = accuracy_score(gts, preds)  # proxy for RGA
    return tot_loss / len(loader.dataset), macro_f1, acc, preds, gts


# ------------------ experiment dict ----------------------
experiment_data = {
    "SPR_BENCH": {
        "Baseline": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
        },
        "MultiTask": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
        },
    }
}


def train_model(name, model, epochs=3, lr=3e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _, _ = run_epoch(model, train_loader, opt)
        val_loss, val_f1, val_acc, pred, gt = run_epoch(model, dev_loader)
        ed = experiment_data["SPR_BENCH"][name]
        ed["metrics"]["train"].append({"epoch": ep, "macro_F1": tr_f1, "RGA": None})
        ed["metrics"]["val"].append({"epoch": ep, "macro_F1": val_f1, "RGA": val_acc})
        ed["losses"]["train"].append({"epoch": ep, "loss": tr_loss})
        ed["losses"]["val"].append({"epoch": ep, "loss": val_loss})
        print(
            f"{name} Epoch{ep}: val_loss={val_loss:.4f} val_F1={val_f1:.3f} RGA={val_acc:.3f} ({time.time()-t0:.1f}s)"
        )
    ed["predictions"] = pred
    ed["ground_truth"] = gt


# ------------------ run both models ----------------------
train_model("Baseline", Baseline())
train_model("MultiTask", MultiTask())

# ------------------ save & exit --------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to working/experiment_data.npy")
