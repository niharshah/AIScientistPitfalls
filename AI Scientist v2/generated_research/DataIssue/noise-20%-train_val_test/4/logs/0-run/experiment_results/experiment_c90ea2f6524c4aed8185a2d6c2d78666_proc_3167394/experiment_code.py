import os, pathlib, random, time, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- experiment data ----------
experiment_data = {
    "SPR_MLM": {
        "metrics": {"train_f1": [], "val_f1": [], "test_f1": None},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
}


# ---------- load SPR_BENCH ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv):
        return load_dataset(
            "csv", data_files=str(root / csv), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


# set your absolute path accordingly
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
print("Dataset sizes:", {k: len(v) for k, v in spr.items()})

# ---------- vocab ----------
special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<MASK>"]
vocab = {tok: i for i, tok in enumerate(special_tokens)}
for seq in spr["train"]["sequence"]:
    for ch in seq:
        if ch not in vocab:
            vocab[ch] = len(vocab)
pad_id, unk_id, cls_id, mask_id = [vocab[t] for t in special_tokens]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq):
    return [cls_id] + [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(lambda x: {"input_ids": encode(x["sequence"])})


# ---------- masking helper ----------
def mask_tokens(inputs, mask_prob=0.15):
    """
    inputs: LongTensor [seq_len]
    Returns masked_inputs, mlm_labels
    """
    inputs = inputs.clone()
    labels = torch.full(inputs.shape, -100, dtype=torch.long)
    for i in range(1, len(inputs)):  # skip CLS token at pos0
        if inputs[i].item() == pad_id:
            break
        if random.random() < mask_prob:
            labels[i] = inputs[i]
            r = random.random()
            if r < 0.8:
                inputs[i] = mask_id
            elif r < 0.9:
                inputs[i] = random.randint(0, vocab_size - 1)
            # else keep original token
    return inputs, labels


# ---------- collate ----------
def collate_fn(batch, do_mask=True):
    seqs = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    if do_mask:
        masked, mlm_labels = zip(*(mask_tokens(s) for s in seqs))
    else:
        masked = seqs
        mlm_labels = [torch.full(s.shape, -100, dtype=torch.long) for s in seqs]

    max_len = max(len(s) for s in masked)
    batch_input = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    batch_attn = torch.zeros_like(batch_input, dtype=torch.bool)
    batch_mlm = torch.full_like(batch_input, -100)
    for i, (inp, attn_mask, mlabel) in enumerate(zip(masked, seqs, mlm_labels)):
        batch_input[i, : len(inp)] = inp
        batch_attn[i, : len(attn_mask)] = True
        batch_mlm[i, : len(mlabel)] = mlabel
    return {
        "input_ids": batch_input,
        "attention_mask": batch_attn,
        "labels": labels,
        "mlm_labels": batch_mlm,
    }


batch_size = 64
train_loader = DataLoader(
    spr["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, do_mask=True),
)
val_loader = DataLoader(
    spr["dev"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, do_mask=False),
)
test_loader = DataLoader(
    spr["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, do_mask=False),
)

num_labels = len(set(spr["train"]["label"]))


# ---------- model ----------
class SPRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=256,
        nhead=8,
        nlayers=6,
        dim_ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(4096, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls_head = nn.Linear(d_model, num_labels)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        cls_repr = x[:, 0, :]
        logits_cls = self.cls_head(cls_repr)
        logits_mlm = self.mlm_head(x)
        return logits_cls, logits_mlm


model = SPRTransformer(vocab_size, num_labels).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# ---------- training helpers ----------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_cls_loss, total_mlm_loss = 0.0, 0.0, 0.0
    preds, gts = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out_cls, out_mlm = model(batch["input_ids"], batch["attention_mask"])
            cls_loss = criterion_cls(out_cls, batch["labels"])
            mlm_loss = criterion_mlm(
                out_mlm.view(-1, vocab_size), batch["mlm_labels"].view(-1)
            )
            loss = cls_loss + 0.5 * mlm_loss if train else cls_loss  # eval ignores mlm
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        total_cls_loss += cls_loss.item() * batch["labels"].size(0)
        total_mlm_loss += mlm_loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(out_cls, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- training loop ----------
epochs = 15
best_val_f1 = 0.0
best_path = os.path.join(working_dir, "best_spr_mlm.pt")

for epoch in range(1, epochs + 1):
    start = time.time()
    train_loss, train_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_f1, _, _ = run_epoch(val_loader, train=False)
    experiment_data["SPR_MLM"]["epochs"].append(epoch)
    experiment_data["SPR_MLM"]["losses"]["train"].append(train_loss)
    experiment_data["SPR_MLM"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_MLM"]["metrics"]["train_f1"].append(train_f1)
    experiment_data["SPR_MLM"]["metrics"]["val_f1"].append(val_f1)
    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_F1={val_f1:.4f}  time={time.time()-start:.1f}s"
    )
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_path)

# ---------- test evaluation ----------
model.load_state_dict(torch.load(best_path))
test_loss, test_f1, test_preds, test_gts = run_epoch(test_loader, train=False)
print(f"TEST Macro-F1 = {test_f1:.4f}")

experiment_data["SPR_MLM"]["test_f1"] = test_f1
experiment_data["SPR_MLM"]["predictions"] = test_preds
experiment_data["SPR_MLM"]["ground_truth"] = test_gts

# ---------- save experiment ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
