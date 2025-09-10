import openai
import time, tiktoken
from openai import OpenAI
import os, anthropic, json
import google.generativeai as genai
import re
import random

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")


def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            try:
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat"]:
                    encoding = tiktoken.encoding_for_model("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(encoding.encode(answer))
                # if print_cost:
                #     print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code


prompt1 = """
Outlined in the following text is the research plan that the machine learning engineer was tasked with building: Title: A Hybrid Neural–Symbolic Model for Robust SPR Classification

Objective:
Develop a simple yet innovative algorithm that, given an L-token sequence (each token comprising a shape glyph from {▲, ■, ●, ◆} and a color from {r, g, b, y}), decides if the sequence adheres to a hidden poly‑factor rule. The rule is a conjunction of atomic predicates from categories such as Shape-Count, Color-Position, Parity, and Order.

Plan Outline:
1. Dataset Design:
   • Synthesize a dataset where each instance is an L-token sequence generated based on a randomly sampled hidden generation rule.
   • The generation rule is constructed by selecting k atomic predicates (e.g., “exactly three ▲”, “4th token is r”, “even number of ■”, or “first ▲ precedes first ●”).
   • Partition the dataset into Train, Dev, and Test splits. The simulation will systematically vary vocabulary sizes, sequence lengths, and rule complexities.

2. Model Architecture:
   A. Transformer Encoder Backbone:
      • Use a lightweight Transformer encoder to obtain token-level representations.
      • Input Embeddings: Create separate embeddings for shapes and colors, then concatenate or sum for combined token representation.
   
   B. Symbolic Rule Induction Module:
      • Aggregate token-level features into interpretable statistics: 
         - Count vectors for shapes
         - Positional embeddings for color features
         - Learned representations capturing parity and order information via simple positional encoding manipulations.
      • Feed these aggregated statistics into a shallow multi-layer perceptron (MLP) that estimates the activation of each atomic predicate.
   
   C. Decision Layer:
      • Combine the predictions from the Transformer and the symbolic module (e.g., through weighted averaging or gating mechanism) to output a binary classification (accept vs. reject).

3. Training Procedure:
   • Loss: Use a binary cross-entropy loss function. Optionally add auxiliary losses to encourage interpretable statistics that align with the known predicate structure.
   • Optimization: Train using Adam with learning-rate scheduling.
   • Training Strategy: Independently train the model on each benchmark’s Train split, tune hyperparameters on the Dev split, and finally evaluate the Test split while comparing accuracy against the SOTA baselines.
   
4. Experiment Details:
   • Vary sequence lengths and rule complexities to test generalization.
   • Baseline Comparisons: 
      - Compare against a vanilla end-to-end Transformer model.
      - Compare with hybrid systems from the literature (e.g., the iterative substitution rule model from arXiv:2306.17249v1).
   • Evaluation Metrics: Report final Test accuracy, rule inference accuracy on synthetic predicates, and robustness over out-of-distribution tests.
   
5. Expected Outcomes and Innovations:
   • A combined neural–symbolic system that not only classifies sequences accurately but also provides interpretable intermediate representations reflective of the underlying predicates.
   • Insights into the balance between end-to-end learning and explicit rule extraction, informed by integrating ideas from symbolic reasoning and sequence modelling literature.
   
By following this plan, we aim to deliver a robust and interpretable SPR solution that can generalize across diverse symbolic rule scenarios.

The following text is the research code that the model produced: 
import os
# Force CPU usage by disabling CUDA devices.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import datasets

# Disable any CUDA-related function to avoid CUDA calls.
# Unconditionally override CUDA stream capturing to avoid calling CUDA functions.
torch.cuda.is_current_stream_capturing = lambda: False

print("Loading SPR_BENCH dataset from CSV files ...")
# -------------------------
# Dataset Loading (Provided Code)
# -------------------------
data_files = {
    "train": "SPR_BENCH/train.csv",
    "dev": "SPR_BENCH/dev.csv",
    "test": "SPR_BENCH/test.csv"
}
spr_dataset = datasets.load_dataset("csv", data_files=data_files)
print("Available splits:", list(spr_dataset.keys()))
print("Sample training instance:", spr_dataset["train"][0])

# To reduce computation time, use subsets (if dataset is large).
def get_subset(ds, max_examples):
    return ds.select(range(min(len(ds), max_examples)))

spr_train = get_subset(spr_dataset["train"], 1000)
spr_dev   = get_subset(spr_dataset["dev"], 200)
spr_test  = get_subset(spr_dataset["test"], 200)

# -------------------------
# Data Preprocessing and Custom Dataset
# -------------------------
# Each sample's "sequence" is a string of tokens.
# Each token is composed of a shape glyph from {▲, ■, ●, ◆} and a color from {r, g, b, y}.
shape2idx = {'▲': 0, '■': 1, '●': 2, '◆': 3}
color2idx = {'r': 0, 'g': 1, 'b': 2, 'y': 3}

class SPRDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = []
        for ex in hf_dataset:
            seq_str = ex["sequence"].strip()
            label = int(ex["label"])
            tokens = seq_str.split()
            token_idxs = []
            for token in tokens:
                if len(token) >= 2:
                    shape = token[0]
                    color = token[1]
                    token_idxs.append([shape2idx.get(shape, 0), color2idx.get(color, 0)])
                else:
                    token_idxs.append([0, 0])
            # Each sample: (tensor of shape [seq_len,2], label as float tensor)
            self.data.append((torch.tensor(token_idxs, dtype=torch.long),
                              torch.tensor(label, dtype=torch.float)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [s.size(0) for s in sequences]
    max_len = max(lengths)
    padded_seqs = []
    for s in sequences:
        pad_size = max_len - s.size(0)
        if pad_size > 0:
            pad = torch.zeros((pad_size, s.size(1)), dtype=torch.long)
            s = torch.cat([s, pad], dim=0)
        padded_seqs.append(s)
    padded_seqs = torch.stack(padded_seqs, dim=0)  # [batch, seq_len, 2]
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_seqs, labels, lengths

train_dataset = SPRDataset(spr_train)
dev_dataset   = SPRDataset(spr_dev)
test_dataset  = SPRDataset(spr_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
dev_loader   = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# -------------------------
# Model Architectures and Hyperparameters
# -------------------------
embedding_dim = 32
hidden_dim = 64
num_transformer_layers = 1
num_heads = 4

num_shapes = len(shape2idx)
num_colors = len(color2idx)

# Transformer Encoder Backbone Model
class TransformerEncoderModel(nn.Module):
    def __init__(self):
        super(TransformerEncoderModel, self).__init__()
        self.shape_embed = nn.Embedding(num_shapes, embedding_dim)
        self.color_embed = nn.Embedding(num_colors, embedding_dim)
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.cls_head = nn.Linear(hidden_dim, 1)
    def forward(self, x, lengths):
        # x: [batch, seq_len, 2]
        shape_idx = x[:,:,0]
        color_idx = x[:,:,1]
        shape_emb = self.shape_embed(shape_idx)   # [batch, seq_len, embedding_dim]
        color_emb = self.color_embed(color_idx)     # [batch, seq_len, embedding_dim]
        token_emb = torch.cat([shape_emb, color_emb], dim=-1)  # [batch, seq_len, embedding_dim*2]
        proj = self.input_proj(token_emb)           # [batch, seq_len, hidden_dim]
        proj = proj.transpose(0, 1)  # Transformer expects [seq_len, batch, hidden_dim]
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        trans_out = self.transformer(proj, src_key_padding_mask=mask)
        trans_out = trans_out.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        # Mean pooling based on true token lengths
        pooled = []
        for i, l in enumerate(lengths):
            pooled.append(trans_out[i, :l, :].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)
        logits = self.cls_head(pooled)
        return logits.squeeze(1)

# Symbolic Rule Induction Module:
# Computes interpretable features: shape counts (4 values) and normalized average positions for colors (4 values).
class SymbolicModule(nn.Module):
    def __init__(self):
        super(SymbolicModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x, lengths):
        batch_size = x.size(0)
        feats = []
        for i in range(batch_size):
            valid_tokens = x[i, :lengths[i], :]  # [l, 2]
            # Count each shape.
            shape_counts = torch.tensor(
                [(valid_tokens[:,0] == s).sum().float() for s in range(num_shapes)],
                device=x.device
            )
            # For each color, compute normalized average position if available.
            color_avgs = []
            for c in range(num_colors):
                inds = (valid_tokens[:,1] == c).nonzero(as_tuple=True)[0]
                if inds.numel() > 0:
                    color_avgs.append(inds.float().mean() / float(lengths[i]))
                else:
                    color_avgs.append(torch.tensor(0.0, device=x.device))
            color_avgs = torch.stack(color_avgs)
            feat = torch.cat([shape_counts, color_avgs], dim=0)  # [8]
            feats.append(feat)
        feats = torch.stack(feats, dim=0)
        logits = self.mlp(feats)
        return logits.squeeze(1)

# Hybrid Neural–Symbolic Model: Combines Transformer-based and Symbolic modules using a learnable gate.
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.transformer_model = TransformerEncoderModel()
        self.symbolic_module = SymbolicModule()
        self.gate = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter.
    def forward(self, x, lengths):
        logit_trans = self.transformer_model(x, lengths)
        logit_sym = self.symbolic_module(x, lengths)
        combined_logit = self.gate * logit_trans + (1 - self.gate) * logit_sym
        return combined_logit

# -------------------------
# Training Setup
# -------------------------
device = torch.device("cpu")
print("Using device:", device)

# Initialize models.
baseline_model = TransformerEncoderModel().to(device)
hybrid_model = HybridModel().to(device)

criterion = nn.BCEWithLogitsLoss()
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
hybrid_optimizer = optim.Adam(hybrid_model.parameters(), lr=1e-3)

# Set seed for reproducibility.
torch.manual_seed(42)
np.random.seed(42)

# Use a single epoch for demonstration to reduce runtime.
num_epochs = 1

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        x, labels, lengths = batch
        x = x.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100.0
    return avg_loss, accuracy

# -------------------------
# Experiment 1: Baseline Transformer Model
# -------------------------
print("\nExperiment 1: Baseline Transformer Model")
print("This experiment trains a baseline Transformer encoder model that uses learned token-level embeddings to classify SPR sequences. "
      "It reports the training loss over the epoch, development set accuracy, and final test accuracy to provide a baseline for comparison.")

baseline_train_losses = []
baseline_dev_accs = []
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = run_epoch(baseline_model, train_loader, baseline_optimizer)
    dev_loss, dev_acc = run_epoch(baseline_model, dev_loader)
    baseline_train_losses.append(train_loss)
    baseline_dev_accs.append(dev_acc)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, Dev Acc = {dev_acc:.2f}%")
test_loss, test_acc = run_epoch(baseline_model, test_loader)
print(f"\nBaseline Transformer Model Final Trainning Accuracy: {train_acc:.2f}%")
print(f"\nBaseline Transformer Model Final Dev Accuracy: {dev_loss:.2f}%")
print(f"\nBaseline Transformer Model Final Test Accuracy: {test_acc:.2f}%")

# Generate Figure_1: Baseline Transformer Training Loss Curve.
plt.figure()
plt.plot(range(1, num_epochs + 1), baseline_train_losses, marker='o', linestyle='-')
plt.title("Figure_1.png: Baseline Transformer Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.grid(True)
plt.savefig("Figure_1.png")
plt.close()

# -------------------------
# Experiment 2: Hybrid Neural–Symbolic Model
# -------------------------
print("\nExperiment 2: Hybrid Neural–Symbolic Model")
print("This experiment trains a Hybrid Neural–Symbolic model that combines a Transformer encoder with a symbolic rule induction module. "
      "The symbolic module computes interpretable features (counts of shapes and normalized color positions) from each SPR sequence. "
      "A learnable gate fuses the outputs of both modules to yield the final binary classification. "
      "This experiment reports training loss, development set accuracy, and final test accuracy for comparison.")

hybrid_train_losses = []
hybrid_dev_accs = []
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = run_epoch(hybrid_model, train_loader, hybrid_optimizer)
    dev_loss, dev_acc = run_epoch(hybrid_model, dev_loader)
    hybrid_train_losses.append(train_loss)
    hybrid_dev_accs.append(dev_acc)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, Dev Acc = {dev_acc:.2f}%")
test_loss_h, test_acc_h = run_epoch(hybrid_model, test_loader)
print(f"\nHybrid Neural–Symbolic Model Final Training Accuracy: {train_acc:.2f}%")
print(f"\nHybrid Neural–Symbolic Model Final Dev Accuracy: {dev_acc_h:.2f}%")
print(f"\nHybrid Neural–Symbolic Model Final Test Accuracy: {test_acc_h:.2f}%")

# Generate Figure_2: Hybrid Model Training Loss and Dev Accuracy Curve.
plt.figure()
plt.plot(range(1, num_epochs + 1), hybrid_train_losses, marker='o', linestyle='-', label="Hybrid Train Loss")
plt.plot(range(1, num_epochs + 1), hybrid_dev_accs, marker='s', linestyle='--', label="Hybrid Dev Accuracy (%)")
plt.title("Figure_2.png: Hybrid Model Training Loss and Dev Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.savefig("Figure_2.png")
plt.close()

# -------------------------
# Final Reporting and SOTA Comparison
# -------------------------
print("\nFinal Report:")
print("Two experiments were conducted on the SPR benchmark (using data subsets for speed):")
print("1) Baseline Transformer: Uses only learned token-level embeddings with a Transformer encoder.")
print("2) Hybrid Neural–Symbolic: Combines the Transformer encoder with a symbolic module that computes interpretable features, "
      "fusing outputs via a learnable gate.")
print(f"Baseline Transformer Test Accuracy: {test_acc:.2f}%")
print(f"Hybrid Neural–Symbolic Test Accuracy: {test_acc_h:.2f}%")
if test_acc <= 0 or test_acc_h <= 0:
    print("Error: One or both models achieved 0% accuracy. Please review the implementation.")
else:
    print("Both models achieved non-zero accuracy, indicating successful training and meaningful results.")

# Save the final models for further analysis if needed.
torch.save(baseline_model.state_dict(), "baseline_model.pth")
torch.save(hybrid_model.state_dict(), "hybrid_model.pth")

The following is the output from the model: Available splits: ['train', 'dev', 'test']
Sample training instance: {'id': 'SPR_train_0', 'sequence': '◆y ■y ■g ■b ■g ●r ▲b ◆g ▲r ▲y ◆b ◆r ●g ◆r ▲y ●y ◆g ■b ●b ▲g ◆r ◆b ■g ●y ▲b ◆g ▲b ◆g ◆y ◆g ■b ◆g', 'label': 0}
Loading SPR_BENCH dataset from CSV files ...
Available splits: ['train', 'dev', 'test']
Sample training instance: {'id': 'SPR_train_0', 'sequence': '◆y ■y ■g ■b ■g ●r ▲b ◆g ▲r ▲y ◆b ◆r ●g ◆r ▲y ●y ◆g ■b ●b ▲g ◆r ◆b ■g ●y ▲b ◆g ▲b ◆g ◆y ◆g ■b ◆g', 'label': 0}
Using device: cpu
"""
propmt2 = """Experiment 1: Baseline Transformer Model
This experiment trains a baseline Transformer encoder model that uses learned token-level embeddings to classify SPR sequences. It reports the training loss over the epoch, development set accuracy, and final test accuracy to provide a baseline for comparison.

Baseline Transformer Model Final Training Accuracy: {a}%
Baseline Transformer Model Final Validation Accuracy: {b}%
Baseline Transformer Model Final Test Accuracy: {c}%

Experiment 2: Hybrid Neural–Symbolic Model
This experiment trains a Hybrid Neural–Symbolic model that combines a Transformer encoder with a symbolic rule induction module. The symbolic module computes interpretable features (counts of shapes and normalized color positions) from each SPR sequence. A learnable gate fuses the outputs of both modules to yield the final binary classification. This experiment reports training loss, development set accuracy, and final test accuracy for comparison.

Hybrid Neural–Symbolic Model Final Training Accuracy: {d}%
Hybrid Neural–Symbolic Model Final Validation Accuracy: {e}%
Hybrid Neural–Symbolic Model Final Test Accuracy: {f}%


Final Report:
Two experiments were conducted on the SPR benchmark (using data subsets for speed):
1) Baseline Transformer: Uses only learned token-level embeddings with a Transformer encoder.
2) Hybrid Neural–Symbolic: Combines the Transformer encoder with a symbolic module that computes interpretable features, fusing outputs via a learnable gate.
Baseline Transformer Training Accuracy: {a}%
Baseline Transformer Validation Accuracy: {b}%
Baseline Transformer Test Accuracy: {c}%
Hybrid Neural–Symbolic Training Accuracy: {d}%
Hybrid Neural–Symbolic Validation Accuracy: {e}%
Hybrid Neural–Symbolic Test Accuracy: {f}%
Both models achieved non-zero accuracy, indicating successful training and meaningful results.

"""

REWARD_MODEL_LLM = "o3-mini"
openai_api_key = "Your OpenAI key"
sys = ( f"You are a professor agent who is serving as an expert reward model that can read a research plan, research code, and code output and are able to determine how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float.\n\n"
        f"You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1 representing how well the model followed the plan, built the code, and got the proper output."
    )

result = []
N = 10 # each instance run 10 times
a,b,c = 85.00, 80.00, 70.00
train, val, test = sorted([round(random.uniform(85, 90), 2) for _ in range(N)]), sorted([round(random.uniform(80, 85), 2) for _ in range(N)]),\
sorted([round(random.uniform(70, 100), 2) for _ in range(N)], reverse=True)
scores = []
for i in range(N):
    prompt = prompt1 + propmt2.format(a=a,b=b,c=c,d=train[i],e=val[i],f=test[i])
    scoring = query_model(
                    model_str=f"{REWARD_MODEL_LLM}",
                    system_prompt=sys,
                    openai_api_key=openai_api_key,
                    prompt=prompt, temp=0.6)

    score = extract_prompt(text=scoring, word="SCORE")
    scores.append(score)
try:
    idx = scores.index(max(scores))
except:
    idx = 0
result.append(idx)
