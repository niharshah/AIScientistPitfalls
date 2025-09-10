import os, pathlib, math, time, json, numpy as np, torch, random
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import DatasetDict, load_dataset


# ---------- reproducibility ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ---------- working / logging ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {"nlayers": {}}  # hyper-parameter tuning container

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- dataset ----------
DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")


def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(fn):
        return load_dataset(
            "csv", data_files=str(root / fn), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(
        train=_load("train.csv"), dev=_load("dev.csv"), test=_load("test.csv")
    )


spr = load_spr_bench(DATA_PATH)
num_labels = len(set(spr["train"]["label"]))
print("Loaded SPR_BENCH with", len(spr["train"]), "train examples.")


# ---------- vocab ----------
def build_vocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in dataset["sequence"]:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
pad_id = vocab["<PAD>"]
unk_id = vocab["<UNK>"]
vocab_size = len(vocab)
print("Vocab size:", vocab_size)


def encode(seq):
    return [vocab.get(c, unk_id) for c in seq]


for split in ["train", "dev", "test"]:
    spr[split] = spr[split].map(
        lambda x: {"input_ids": encode(x["sequence"])}, remove_columns=[]
    )


# ---------- collate ----------
def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(t.size(0) for t in input_ids)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(padded, dtype=torch.bool)
    for i, t in enumerate(input_ids):
        padded[i, : t.size(0)] = t
        attn[i, : t.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


batch_size = 128
loaders = {
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
        self.pos_emb = nn.Parameter(torch.randn(5000, d_model) * 0.02)  # max len 5000
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pos_emb[:seq_len]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            1, keepdim=True
        )
        return self.classifier(x)


# ---------- training & evaluation helpers ----------
criterion = nn.CrossEntropyLoss()


def run_epoch(model, optimizer, split):
    is_train = split == "train"
    model.train() if is_train else model.eval()
    total_loss, preds, gts = 0.0, [], []
    for batch in loaders[split]:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
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
    return (
        total_loss / len(loaders[split].dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# ---------- hyper-parameter sweep ----------
nl_values = [2, 4, 6, 8]
epochs = 10
for nl in nl_values:
    print(f"\n===== Training with nlayers={nl} =====")
    key = f"SPR_BENCH_n{nl}"
    experiment_data["nlayers"][key] = {
        "metrics": {"train_f1": [], "val_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }

    model = CharTransformer(vocab_size, num_labels, nlayers=nl).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best_f1 = 0.0
    best_path = os.path.join(working_dir, f"best_model_n{nl}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1, _, _ = run_epoch(model, optimizer, "train")
        val_loss, val_f1, _, _ = run_epoch(model, optimizer, "dev")

        exp = experiment_data["nlayers"][key]
        exp["epochs"].append(epoch)
        exp["losses"]["train"].append(tr_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train_f1"].append(tr_f1)
        exp["metrics"]["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    # test evaluation
    model.load_state_dict(torch.load(best_path))
    test_loss, test_f1, test_preds, test_gts = run_epoch(
        model, optimizer=None, split="test"
    )
    experiment_data["nlayers"][key]["predictions"] = test_preds
    experiment_data["nlayers"][key]["ground_truth"] = test_gts
    print(f"best val_F1={best_f1:.4f} | Test_F1={test_f1:.4f}")

    # cleanup to free GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

# ---------- save experiments ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
