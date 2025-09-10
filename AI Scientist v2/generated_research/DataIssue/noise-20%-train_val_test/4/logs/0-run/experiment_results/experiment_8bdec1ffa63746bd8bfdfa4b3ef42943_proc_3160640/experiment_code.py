import os, pathlib, math, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict

# ---------- reproducibility ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment data ----------
experiment_data = {
    "weight_decay": {"SPR_BENCH": {}}  # will hold one sub-dict per wd value (as str)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- dataset loader ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    from datasets import load_dataset

    def _load(file):
        return load_dataset(
            "csv", data_files=str(root / file), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)
num_labels = len(set(spr["train"]["label"]))
print("Loaded SPR_BENCH with", len(spr["train"]), "train examples.")


# ---------- build vocab ----------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id, unk_id, vocab_size = vocab["<PAD>"], vocab["<UNK>"], len(vocab)
print("Vocab size:", vocab_size)


# ---------- encode sequences ----------
def encode(seq):
    return [vocab.get(ch, unk_id) for ch in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------- collate ----------
def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(x.size(0) for x in input_ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, seq in enumerate(input_ids):
        padded[i, : seq.size(0)] = seq
        attn[i, : seq.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders_full = {
    split: DataLoader(
        spr[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )
    for split in ["train", "dev", "test"]
}


# ---------- model ----------
class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        d_model=128,
        nhead=4,
        nlayers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)  # max len 5k
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


# ---------- training / evaluation helpers ----------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, optimizer, loaders, split):
    is_train = split == "train"
    model.train() if is_train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loaders[split]:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / len(loaders[split].dataset)
    f1 = f1_score(gts, preds, average="macro")
    return avg_loss, f1, preds, gts


# ---------- hyperparameter sweep ----------
weight_decays = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
epochs = 10
global_best_f1, global_best_state = -1.0, None
global_best_wd = None

for wd in weight_decays:
    print(f"\n=== Training with weight_decay={wd} ===")
    # init containers
    exp_run = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "epochs": list(range(1, epochs + 1)),
    }
    # init model/optimizer
    model = CharTransformer(vocab_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
    best_val_f1_this_run, best_state_this_run = -1.0, None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, optimizer, loaders_full, "train")
        val_loss, val_f1, _, _ = run_epoch(model, optimizer, loaders_full, "dev")
        exp_run["metrics"]["train_f1"].append(tr_f1)
        exp_run["metrics"]["val_f1"].append(val_f1)
        exp_run["losses"]["train"].append(tr_loss)
        exp_run["losses"]["val"].append(val_loss)
        print(
            f"  Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1_this_run:
            best_val_f1_this_run = val_f1
            best_state_this_run = {k: v.cpu() for k, v in model.state_dict().items()}

    # save per-weight-decay run data
    experiment_data["weight_decay"]["SPR_BENCH"][str(wd)] = exp_run

    # update global best
    if best_val_f1_this_run > global_best_f1:
        global_best_f1 = best_val_f1_this_run
        global_best_state = best_state_this_run
        global_best_wd = wd

# ---------- test evaluation with best model ----------
print(f"\nBest dev F1={global_best_f1:.4f} achieved with weight_decay={global_best_wd}")
best_model = CharTransformer(vocab_size, num_labels).to(device)
best_model.load_state_dict(global_best_state)
best_model.to(device)
test_loss, test_f1, test_preds, test_gts = run_epoch(
    best_model, None, loaders_full, "test"
)
print(f"Test MacroF1: {test_f1:.4f}")

# store predictions & ground truth of best run
experiment_data["weight_decay"]["SPR_BENCH"]["best_wd"] = global_best_wd
experiment_data["weight_decay"]["SPR_BENCH"]["test_f1"] = test_f1
experiment_data["weight_decay"]["SPR_BENCH"]["predictions"] = test_preds
experiment_data["weight_decay"]["SPR_BENCH"]["ground_truth"] = test_gts

# ---------- save all experiment data ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
