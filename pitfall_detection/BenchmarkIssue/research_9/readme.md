
# Neural-Grammar-Symbolic Models in Symbolic Pattern Recognition

## Overview
This repository contains the implementation of a neural-grammar-symbolic model designed to tackle Symbolic Pattern Recognition (SPR) tasks. The SPR task centers on determining whether a sequence of abstract symbols adheres to a hidden target rule, challenging due to the symbolic nature that requires a blend of neural perception, grammar parsing, and symbolic reasoning.

## Key Features
- **Neural-Grammar-Symbolic Model:** Integrates neural networks with grammar and symbolic reasoning to enhance symbolic pattern recognition.
- **Multi-Head Attention Mechanisms:** Focuses on key sequence components to facilitate reasoning across atomic predicate categories like Shape-Count, Color-Position, Parity, and Order.
- **Synthetic Dataset:** Created for testing, consisting of sequences of varying lengths with diverse logical rules.
- **Comparative Analysis:** Evaluates model performance against traditional machine learning models like Random Forests and Decision Trees.

## Implementation
The neural-grammar-symbolic model involves the following components:
- **Attention Mechanism:** Enhances model’s focus on crucial sequence parts using multi-head attention.
- **Embedding Strategies:** Embeds each symbol using positional encodings to capture sequence order.
- **Feed-Forward Network:** Outputs predictions by processing features obtained from multi-head attention.

### Model Architecture
- **Attention Formula:**
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]
- **Positional Encodings:**
  \[
  \text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  \]
  \[
  \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  \]
- **Prediction Equation:**
  \[
  \hat{y} = \sigma(W_h \cdot \phi(\mathbf{x}) + b_h)
  \]

### Dataset
- Synthetic data set includes sequences from four categories: Shape-Count, Color-Position, Parity, and Order.
- Split into training, validation, and test sets in the ratio of 70:15:15.

## Results
- Accuracy of the neural-grammar-symbolic model: 47.8%
- Compares with accuracies from Random Forests at 50.7% and Decision Trees at 49.7%.

## Future Work
- **Refinement Areas:** Improve attention mechanisms, embedding strategies, dataset complexity.
- **Model Improvements:** Enhance symbolic reasoning capabilities and integrate more structured domain knowledge.

## Getting Started
- Clone the repository:
  ```bash
  git clone <repository-url>
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the model:
  ```bash
  python main.py
  ```

## License
This project is licensed under the MIT License.

---

Please refer to the project documentation for more details on each component and a complete guide on reproducing the experiments.
```
