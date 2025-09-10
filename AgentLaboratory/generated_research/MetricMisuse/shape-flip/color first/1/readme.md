# Neural-Symbolic Transformer for Symbolic Pattern Recognition (SPR)

Welcome to the repository for the R-NSR project, a neural-symbolic framework that unifies a lightweight transformer encoder with a sparse concept extraction layer and a differentiable symbolic reasoning module designed specifically for the Symbolic Pattern Recognition (SPR) task.

## Overview

The goal of the R-NSR model is to determine whether a given sequence of abstract tokens—each token encoding features such as shape and color—satisfies hidden poly-factor rules. Our architecture integrates:

- **Transformer Encoder:** A lightweight transformer with self-attention that produces rich contextual representations from input sequences.
- **Sparse Concept Extraction Layer:** A layer that distills dense representations into interpretable, sparse predicate activations corresponding to four distinct rule types. L1 regularization encourages near-binary activations.
- **Differentiable Symbolic Reasoning Module:** A soft logical reasoning module that combines the predicate activations via a product-based soft AND operation, yielding a final binary decision.

This integrated approach supports both high performance and interpretability in low-data regimes.

## Repository Structure

- **README.md** – This file.
- **/src** – Source code for model implementation.
  - `model.py` – Contains the definition of the R-NSR model including the transformer encoder, sparse extraction layer, and symbolic reasoning module.
  - `train.py` – Training script using Adam optimizer and scheduled learning rate decay.
  - `utils.py` – Utility functions for data preprocessing, dataset construction, and evaluation.
- **/data** – Contains generation scripts and sample data for the synthetic SPR dataset.
- **/results** – Contains logs, training plots, and comparative bar charts for ablation studies.
- **/experiments** – Scripts for ablation studies and further experiments.

## Model Architecture

### Transformer Encoder
- Processes input sequences of tokens (each token encoding shape and color features).
- Uses positional embeddings to handle variable-length inputs.
- Captures long-range dependencies using a self-attention mechanism.

### Sparse Concept Extraction Layer
- Applies a linear transformation with a sigmoid activation to generate four candidate predicate scores.
- Enforces sparsity via an L1 regularization penalty.
- Predicates directly map to human-understandable rules (e.g., “exactly two circles”).

### Differentiable Symbolic Reasoning Module
- Integrates the predicates through an element-wise product with a learnable bias.
- Mimics a logical AND operation while remaining fully differentiable via a sigmoid non-linearity.
- Final prediction is computed as:  
  ŷ = σ(∏₍ⱼ₌₁⁴₎ sᵢⱼ + b),  
  where sᵢⱼ are predicate activations for sample i.

## Training Details

- **Loss Function:** Binary cross-entropy loss augmented with an L1 penalty on the sparse component.
  
  Loss:  L_total = BCE(y, ŷ) + λ ∑ᵢ∑ⱼ |sᵢⱼ|
  
- **Optimizer:** Adam with a scheduled learning rate decay.
- **Training Regime:** Experiments were conducted in a low-data setting (500 training examples) using CPU-only training for one epoch.
- **Batch Size:** 32

## Dataset

The SPR dataset is synthetically generated:
- Tokens represent an abstract symbol as a two-character string:
  - First character: Shape (e.g., triangle, square, circle, lozenge)
  - Second character: Color (r, g, b, y)
- Variable sequence lengths (up to 30 tokens).
- Labels indicate if the sequence follows a hidden poly-factor rule generated from four predicate types (e.g., shape count, color consistency, positional requirements, ordering constraints).

Data splits:
- **Training:** 500 examples
- **Development:** Used for hyperparameter tuning and early stopping.
- **Test:** For final evaluation.

## Evaluation Metrics

The performance is measured using:
- **Standard Test Accuracy**
- **Color-Weighted Accuracy (CWA)**
- **Shape-Weighted Accuracy (SWA)**

Our full R-NSR model achieved:
- Test Accuracy: 56.52%
- CWA: 56.61%
- SWA: 55.32%

In comparison, a baseline transformer model (without sparse and symbolic modules) reached 46.99% accuracy, indicating an improvement of nearly 10 percentage points.

## Ablation Study

We conducted ablation studies to validate the importance of the sparse extraction and symbolic reasoning modules. Removing these modules resulted in:
- Lower test accuracy
- Reduced interpretability of the model outputs

Figures and training dynamics are available in the `/results` folder for further inspection.

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/neural-symbolic-spr.git
   cd neural-symbolic-spr
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   *Note: The requirements file includes libraries such as PyTorch, NumPy, and Matplotlib.*

### Running the Model

- **Training:**
  ```
  python src/train.py --data_path ./data/spr_dataset.json --epochs 1 --batch_size 32
  ```

- **Evaluation:**
  ```
  python src/evaluate.py --data_path ./data/spr_test.json --model_path ./results/rnsr_model.pth
  ```

- **Ablation Study:**
  ```
  python experiments/ablation_experiment.py --config experiments/config_ablation.json
  ```

## Interpretation and Visualization

Post-training, you can inspect the sparse predicate activations to directly map the model’s decisions to logical rules (e.g., “the first square precedes the first triangle”). Visualization scripts are available in the `/results` folder to help you plot training loss curves and predicate activations.

## Future Work

Future developments include:
- Scaling the dataset and model architecture with deeper transformer encoders.
- Exploring advanced structured sparsity techniques and alternative differentiable logic operations.
- Applying the framework to domain-specific applications in finance, healthcare, and robotics.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas ahead of time.

## License

This project is licensed under the MIT License.

## Contact

For further information or any questions regarding this project, please contact [your email] or open an issue in the repository.

Happy Researching!