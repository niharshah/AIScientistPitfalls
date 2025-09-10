import os, pathlib, random, math, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------------- basic setup --------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- experiment tracker -------
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": []},
        "losses": {"train": [], "val": [], "test": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# -------------- data loading ---------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# Allow an environment variable override so the script is portable
DATA_PATH = pathlib.Path(
    os.getenv("SPR_DATA", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
)
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"SPR_BENCH folder not found at {DATA_PATH}. "
        "Set env variable SPR_DATA to the correct path."
    )

spr = load_spr_bench(DATA_PATH)

# -------------- vocabulary build ------------
special_tokens = ["<PAD>", "<UNK>", "<MASK>"]
vocab = {tok: i for i, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
pad_id, unk_id, mask_id = vocab["<PAD>"], vocab["<UNK>"], vocab["<MASK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ------------- dataloader ------------------
def collate(batch):
    ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(s) for s in ids)
    padded = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = seq
        attn[i, : len(seq)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders = {
    sp: DataLoader(
        spr[sp], batch_size=batch_size, shuffle=(sp == "train"), collate_fn=collate
    )
    for sp in ["train", "dev", "test"]
}
num_labels = len(set(spr["train"]["label"]))


# -------------- model -----------------------
class SPRTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 256
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.randn(4096, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=6)
        self.cls_head = nn.Linear(d_model, num_labels)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos[:seq_len]
        x = self.enc(x, src_key_padding_mask=~attention_mask)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        logits_cls = self.cls_head(pooled)
        logits_mlm = self.mlm_head(x)
        return logits_cls, logits_mlm


model = SPRTransformer().to(device)


# -------------- helper : masking ------------
def mask_tokens(inp_ids, mask_prob: float = 0.15):
    """
    Create masked-language-model style inputs and labels.
    Bugfix: torch.randint now receives a tuple size constructed from an int,
            and the corner-case of zero random positions is handled.
    """
    inp = inp_ids.clone()
    labels = torch.full_like(inp, -100)

    # Select positions to mask
    candidate_mask = inp != pad_id
    probs = torch.rand(inp.shape, device=inp.device)
    mask_selector = candidate_mask & (probs < mask_prob)
    labels[mask_selector] = inp[mask_selector]

    # Replace 80% with <MASK>
    rand = torch.rand(inp.shape, device=inp.device)
    replace_mask = mask_selector & (rand < 0.8)
    inp[replace_mask] = mask_id

    # Replace 10% with random token
    random_mask = mask_selector & (rand >= 0.8) & (rand < 0.9)
    num_random = int(random_mask.sum().item())
    if num_random > 0:
        random_tokens = torch.randint(3, vocab_size, (num_random,), device=inp.device)
        inp[random_mask] = random_tokens

    # 10% keep original (handled implicitly)
    return inp, labels


# -------------- training / eval -------------
cls_criterion = nn.CrossEntropyLoss()
mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs, best_val = 20, 0.0
early_stops = 4
no_improve = 0
for epoch in range(1, epochs + 1):
    # --- train ---
    model.train()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loaders["train"]:
        batch = {k: v.to(device) for k, v in batch.items()}
        masked_inp, mlm_labels = mask_tokens(batch["input_ids"])
        logits_cls, logits_mlm = model(masked_inp, batch["attention_mask"])
        loss_cls = cls_criterion(logits_cls, batch["labels"])
        loss_mlm = mlm_criterion(logits_mlm.view(-1, vocab_size), mlm_labels.view(-1))
        loss = loss_cls + 0.5 * loss_mlm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss_cls.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits_cls, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    train_loss = tot_loss / len(loaders["train"].dataset)
    train_f1 = f1_score(gts, preds, average="macro")

    # --- validate ---
    model.eval()
    with torch.no_grad():
        val_loss, preds, gts = 0.0, [], []
        for batch in loaders["dev"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_cls, _ = model(batch["input_ids"], batch["attention_mask"])
            loss_cls = cls_criterion(logits_cls, batch["labels"])
            val_loss += loss_cls.item() * batch["labels"].size(0)
            preds.extend(torch.argmax(logits_cls, 1).cpu().tolist())
            gts.extend(batch["labels"].cpu().tolist())
        val_loss /= len(loaders["dev"].dataset)
        val_f1 = f1_score(gts, preds, average="macro")

    # log
    experiment_data["SPR_BENCH"]["epochs"].append(epoch)
    experiment_data["SPR_BENCH"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_BENCH"]["metrics"]["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"train_F1={train_f1:.4f} val_F1={val_f1:.4f}"
    )

    # early stopping
    if val_f1 > best_val:
        best_val = val_f1
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(working_dir, "best_model.pt"))
    else:
        no_improve += 1
        if no_improve >= early_stops:
            print("Early stopping.")
            break

# --------------- test -----------------------
model.load_state_dict(
    torch.load(os.path.join(working_dir, "best_model.pt"), map_location=device)
)
model.to(device)
model.eval()
with torch.no_grad():
    preds, gts, test_loss = [], [], 0.0
    for batch in loaders["test"]:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits_cls, _ = model(batch["input_ids"], batch["attention_mask"])
        loss_cls = cls_criterion(logits_cls, batch["labels"])
        test_loss += loss_cls.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits_cls, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    test_loss /= len(loaders["test"].dataset)
    test_f1 = f1_score(gts, preds, average="macro")
    print(f"TEST MacroF1: {test_f1:.4f}")

experiment_data["SPR_BENCH"]["losses"]["test"] = test_loss
experiment_data["SPR_BENCH"]["metrics"]["test_f1"].append(test_f1)
experiment_data["SPR_BENCH"]["predictions"] = preds
experiment_data["SPR_BENCH"]["ground_truth"] = gts

# --------------- save -----------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
