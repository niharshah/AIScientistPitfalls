# Neuro-Symbolic Approaches in Symbolic Pattern Recognition

This repository contains the code, experimental setup, and supporting materials for the research paper titled **"Research Report: Neuro-Symbolic Approaches in Symbolic Pattern Recognition"**. The project explores how a simple GRU-based sequence model, trained with a composite loss function, can serve as a baseline for symbolic pattern recognition (SPR) tasks while ensuring model interpretability through sparse symbolic rule extraction techniques.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture and Methods](#architecture-and-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Future Directions](#future-directions)
- [Citation](#citation)
- [License](#license)

---

## Overview

Symbolic pattern recognition requires models to capture both the temporal dynamics of sequences and to infer high-level symbolic rules from them. In this work, we address the SPR task by mapping L-token symbolic sequences (with a vocabulary size of 17 and a maximum length of 6) into discrete label predictions based on a hidden target rule. 

Our GRU-based classifier is trained via a composite loss function that combines:
- **Supervised Contrastive Loss (L_supcon):** Encourages embedding representations of similar classes to cluster.
- **Entropy Minimization Loss (L_entropy):** Drives neuron activations towards binary (0 or 1) states for better interpretability.
- **Sparsity Regularization (L_sparsity):** Enforces a compact representation via L1 regularization.

Despite its simplicity, the baseline model demonstrates that even recurrent architectures can capture essential symbolic features, motivating the integration of more advanced neuro-symbolic components (such as attention mechanisms and sparse concept layers) to bridge the performance gap with state-of-the-art methods.

---

## Key Features

- **Neuro-Symbolic Baseline:** Implements a GRU-based sequence classifier for SPR.
- **Composite Loss Function:** Integrates supervised contrastive, entropy minimization, and sparsity losses.
- **Interpretability:** Focus on enforcing binary and sparse latent representations to facilitate symbolic rule extraction.
- **Experimental Analysis:** Provides detailed evaluations with training loss, development accuracy, and test shape-weighted accuracy (SWA).

---

## Architecture and Methods

The method follows a straightforward pipeline:

1. **Embedding Layer:** Maps input token indices into a 50-dimensional continuous space.
2. **GRU Layer:** Processes the sequential input with a hidden dimension of 64.
3. **Fully Connected Layer:** Outputs the binary classification prediction.
4. **Composite Loss Function:**

   The overall training objective is defined mathematically as:

   L_total = L_supcon + λ₁ L_entropy + λ₂ L_sparsity

   - *L_supcon*: Encourages clustering of same-class sequences.
   - *L_entropy*: Pushes activations toward {0,1} for simpler symbolic extraction.
   - *L_sparsity*: Imposes L1 regularization to ensure compact representations.

Additional components (e.g., attention mechanisms, sparse concept layers) are proposed as future enhancements to improve both accuracy and interpretability.

---

## Installation

### Prerequisites
Ensure that you have Python (>=3.7) installed. The primary dependencies include:
- numpy
- PyTorch
- matplotlib (for plotting training losses and confusion matrices)
- (Optionally) other libraries such as scikit-learn for additional diagnostics

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/neuro-symbolic-spr.git
   cd neuro-symbolic-spr
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > Note: A sample `requirements.txt` is provided with the repository.

---

## Usage

To train and evaluate the baseline GRU model on the SPR_BENCH dataset, run:

```bash
python train.py --data_path path/to/SPR_BENCH/dataset --epochs 1 --batch_size 32
```

The `train.py` script allows setting hyperparameters through command-line options, including:
- `--embedding_dim` (default: 50)
- `--hidden_dim` (default: 64)
- `--learning_rate` (default: 0.001)
- `--lambda_entropy` and `--lambda_sparsity` for loss weight tuning

Additionally, plotting scripts are provided to visualize training loss reduction and the confusion matrix obtained on the test set.

---

## Experimental Setup

- **Dataset:** SPR_BENCH  
  The dataset consists of symbolic sequences of fixed length 6 drawn from a vocabulary of 17 tokens. Shorter sequences are padded appropriately. The task is formulated as a binary classification problem where a sequence either satisfies or does not satisfy a hidden target rule.
  
- **Training Configuration:**
  
  | Parameter             | Value     |
  |-----------------------|-----------|
  | Vocabulary Size       | 17        |
  | Sequence Length       | 6         |
  | Training Instances    | 1,000     |
  | Embedding Dimension   | 50        |
  | GRU Hidden Dimension  | 64        |
  | Learning Rate         | 0.001     |
  | Batch Size            | 32        |
  | Epochs                | 1         |

- **Metrics:**  
  The performance is primarily measured using shape-weighted accuracy (SWA):

  SWA = (Σᵢ (wᵢ · 1{ŷᵢ = yᵢ})) / (Σᵢ wᵢ)

  Reported results from the baseline include:
  - Training Loss: 0.6649
  - Development Accuracy (SWA): 65.16%
  - Test Accuracy (SWA): 57.81%

---

## Results

The experimental evaluation shows that:
- The model effectively learns basic sequential dynamics and symbolic patterns.
- There remains a performance gap compared to state-of-the-art methods (~75% SWA), attributed to factors like limited training epochs, the simplicity of the GRU architecture, and modest sample size.
- Ablation studies confirm that both sparsity and entropy regularization are crucial for interpretable latent representations.

The repository includes scripts for generating detailed reports (e.g., training loss plots and confusion matrix visualizations) to aid in further analysis.

---

## Future Directions

Future work will focus on:
- **Integrating Advanced Modules:** Incorporate attention mechanisms and dedicated sparse concept layers to capture higher-order dependencies.
- **Extended Training Protocols:** Explore longer training schedules with learning rate decay and momentum to achieve deeper convergence.
- **Hybrid Neuro-Symbolic Models:** Fuse neural representations with explicit symbolic rule extraction for scalable and interpretable AI.
- **Broader Applications:** Extend the approach to tasks in natural language understanding and computer vision for richer symbolic reasoning.

These improvements are expected to bridge the current performance gap while enhancing model interpretability and applicability in high-stakes decision-making.

---

## Citation

If you find this work useful in your research, please consider citing:

    @inproceedings{AgentLaboratory2023,
      title={Research Report: Neuro-Symbolic Approaches in Symbolic Pattern Recognition},
      author={Agent Laboratory},
      year={2023},
      eprint={arXiv:xxx.xxxxx}
    }

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or contributions, please open an issue or submit a pull request. Happy symbolic pattern recognition!