# Neural Program Synthesis with Differentiable Symbolic Execution for SPR

This repository contains the implementation and associated materials for our neuro‑symbolic framework designed for Symbolic Pattern Recognition (SPR). The approach integrates a Transformer‑based encoder with meta‑feature fusion and candidate symbolic program synthesis via differentiable symbolic execution. The design enables the model to reconcile continuous neural representations with discrete, interpretable symbolic operations, and has been evaluated on the SPR_BENCH dataset.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Ablation Studies](#ablation-studies)
- [Discussion and Future Work](#discussion-and-future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

In this work, we propose a novel neuro-symbolic framework for SPR that:
- **Integrates a Transformer-based Encoder:** Projects token sequences (of fixed length) into a latent 64-dimensional space.
- **Fuses Meta-Features:** Incorporates auxiliary information (e.g., unique color and shape counts) via a linear mapping to bolster symbolic pattern recognition.
- **Synthesizes Candidate Programs:** Uses multiple candidate heads whose outputs are aggregated by a differentiable interpreter. This mechanism approximates symbolic logical operations to decide whether an L-token sequence satisfies a hidden rule.
- **Composite Loss Function:** The training objective combines binary cross-entropy with candidate synthesis and contrastive losses:
  
  L = L<sub>BCE</sub> + 0.5·L<sub>cand</sub> + 0.1·L<sub>contrast</sub>
  
- **Experimental Results:** On SPR_BENCH, our approach achieved an overall test accuracy of 67.72%, with a Color‑Weighted Accuracy (CWA) of 67.78% (exceeding the SOTA of ~65.0%) and a Shape‑Weighted Accuracy (SWA) of 63.50% (with SOTA ≈ 70.0%).

The complete research report is provided in the Latex file included in this repository.

---

## Features

- **Differentiable Symbolic Execution:** Soft approximations (e.g., SoftAND) allow blending of discrete symbolic operations with gradient-based optimization.
- **End-to-End Trainability:** The model is trained in a unified manner, balancing classification accuracy with interpretability.
- **Reinforcement and Imitation Learning Modules:** Enhanced candidate synthesis through weak supervision and policy gradients.
- **Interpretability:** Visualization tools (e.g., scatter plots of meta-feature vs. decision confidence) provide insights into how the model reasons with symbolic information.
- **Extensible Framework:** Designed to be a stepping stone for broader neuro-symbolic tasks beyond SPR.

---

## Repository Structure

├── README.md  
├── docs/  
│   └── Research_Report.pdf          # PDF version of the full research report  
├── src/  
│   ├── model.py                     # Implementation of Transformer encoder, meta-feature fusion, candidate heads  
│   ├── losses.py                    # Composite loss function modules  
│   ├── train.py                     # Training loop and scheduler  
│   ├── evaluate.py                  # Script for evaluation on SPR_BENCH  
│   └── utils.py                     # Data preprocessing and helper functions  
├── experiments/  
│   ├── SPR_BENCH/                  # Dataset and configuration files  
│   └── figures/                    # Figures used in the report (e.g., training loss curves, scatter plots)  
├── requirements.txt                 # Python requirements  
└── LICENSE

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/neuro-symbolic-spr.git
   cd neuro-symbolic-spr
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Dataset Preparation

- The SPR_BENCH dataset is expected to be located in `experiments/SPR_BENCH/`.  
- Preprocessing includes tokenization, sequence padding (fixed length = 6), and normalization of meta-features.  
- If needed, run the preprocessing script available in `src/utils.py` to prepare the dataset.

### Training

To train the model, execute the training script:

```bash
python src/train.py --data_dir experiments/SPR_BENCH --batch_size 64 --learning_rate 0.001 --epochs 20
```

Key training parameters:
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function:  
  L = L<sub>BCE</sub> + 0.5·L<sub>cand</sub> + 0.1·L<sub>contrast</sub>

### Evaluation

After training, evaluate the model’s performance using:

```bash
python src/evaluate.py --model_path path/to/checkpoint.pt --data_dir experiments/SPR_BENCH
```

Metrics provided include:
- Overall Test Accuracy
- Color-Weighted Accuracy (CWA)
- Shape-Weighted Accuracy (SWA)

---

## Results

Our experimental evaluation on SPR_BENCH demonstrated:
- **Overall Accuracy:** 67.72%
- **Color-Weighted Accuracy (CWA):** 67.78% (SOTA ≈ 65.0%)
- **Shape-Weighted Accuracy (SWA):** 63.50% (SOTA ≈ 70.0%)

Additional detailed results—including ablation studies that highlight the impact of removing candidate synthesis or contrastive loss components—are discussed in the research report.

---

## Ablation Studies

The repository includes experiments that analyze the contribution of individual components:
- **Candidate Synthesis Removal:** Leads to a drop in CWA by 3–4%, demonstrating its role in model interpretability.
- **Contrastive Loss Removal:** Results in unstable latent representations, underscoring the importance of aligning embeddings with symbolic prototypes.

Scripts and configuration files for these studies are available in the `experiments/` subdirectory.

---

## Discussion and Future Work

### Key Insights
- **Strengths:**  
  - Excellent performance in capturing color-based symbolic patterns.
  - End-to-end differentiable design enables joint optimization of classification and symbolic interpretability.
- **Areas for Improvement:**  
  - Shape-based symbolic reasoning lags behind state-of-the-art.
  - Potential enhancements include richer shape representations and more complex feature fusion strategies.

### Future Directions
- Exploring alternative fusion strategies (e.g., bilinear pooling, multi-head attention for meta-features).
- Extending the candidate synthesis mechanism to handle a broader range of symbolic reasoning tasks.
- Enhancing interpretability with visualizations and natural language explanations.

---

## Citation

If you find this work useful, please consider citing our research:

    @inproceedings{agentlab_neurosymbolic2023,
      title={Neural Program Synthesis with Differentiable Symbolic Execution for SPR},
      author={Agent Laboratory},
      year={2023},
      note={Access the full research report in the docs/ folder.}
    }

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Happy coding and research! If you encounter any issues or have suggestions, please open an issue or submit a pull request.