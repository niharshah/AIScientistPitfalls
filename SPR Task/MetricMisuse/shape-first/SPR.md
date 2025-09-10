# Title: Develop a Robust Algorithm for a Novel Task “Symbolic Pattern Reasoning” 

## Background

**Symbolic Pattern Reasoning (SPR)**

* SPR  is a classification task distilled from complex reasoning patterns found in various real-world domains such as finance, academic publishing, and scientific discovery, where latent symbolic rules govern decision-making. Solving this task has the potential to greatly impact automated reasoning systems in practice. In SPR, each instance consists of a symbolic sequence $S = [s_1, s_2, \dots, s_L]$ of length $L$, where each token $s_i$ is composed of an abstract shape glyph from the set **{▲, ■, ●, ◆}** and a color glyph from the set **{r, g, b, y}**.


* A hidden *generation rule* $R$ governs the mapping from an input sequence $S$ to a binary label: *accept / reject*. This rule encapsulates a logical structure that governs how different symbol combinations should be interpreted, leading to the classification decision.

* The rules in SPR are **poly‑factor**, meaning each rule is the result of applying a logical AND across $k$ atomic predicates. These atomic predicates are derived from the following categories:
  1. **Shape-Count**: Refers to conditions based on the frequency of a specific shape within the sequence. For instance, a predicate could specify "exactly three ▲," meaning that the rule only holds if there are exactly three occurrences of the shape ▲.

  2. **Color-Position**: Conditions based on the color of a specific token at a defined position in the sequence. An example predicate might be "token 4 is **r**," meaning the fourth token in the sequence must be colored red for the rule to hold.

  3. **Parity**: Conditions involving the even or odd count of specific shapes or colors. For example, "the number of ■ is even" could be a rule, meaning that the total count of squares ■ in the sequence must be an even number for the sequence to be accepted.

  4. **Order**: Relational conditions on the order of specific tokens in the sequence. An example could be "the first ▲ precedes the first ●," meaning that the first occurrence of the shape ▲ must appear before the first occurrence of the shape ● in the sequence.

By solving the SPR task, we aim to unlock the ability to automatically identify and classify complex symbolic sequences that follow hidden, intricate rules. This has significant potential for automation in various domains where symbolic data patterns need to be understood, such as automating financial analysis, enhancing academic publishing workflows, or improving decision-making systems in complex environments.


## SPR_BENCH Benchmark from HuggingFace

In this experiment, we usethe SPR_BENCH benchmark sourced from HuggingFace to evaluate the performance of machine learning models on a symbolic pattern recognition task. The benchmarks include a variety of symbolic patterns that challenge models to classify sequences of abstract symbols under varying conditions. SPR_BENCH benchmark is standardized with the following fixed global parameters:


| Field                         | Value                                    |
| ----------------------------- | ---------------------------------------- |
| **Instances per split**       | Train = 20,000 Dev = 5,000 Test = 10,000 |
| **Label balance**             | 50 % accept / 50 % reject                |
| **Public leaderboard metric** | Label Accuracy on each benchmark         |


### Metrics

The evaluation is based on two distinct complexity dimensions for any given sequence S:
* **Shape Complexity** $C_s(S_i)$: The number of **unique** shape glyphs {▲, ■, ●, ◆} in the sequence. (Range: 1-4)  
* **Color Complexity**  $C_c(S_i)$: The number of **unique** color glyphs {r, g, b, y} in the sequence. (Range: 1-4)

#### **Metric 1: Shape-Weighted Accuracy (SWA)**

* **Description**: This metric heavily prioritizes performance on instances with high shape variety.  
* $\text{SWA} = \frac{\sum_{i=1}^{N} w_i^{\text{SWA}} \cdot \mathbb{I}(y_i = \hat{y}_i)}{\sum_{i=1}^{N} w_i^{\text{SWA}}}, \quad \text{where } w_i^{\text{SWA}} = C_s(S_i)$
* **Rationale (as presented to the AI agent)**: SWA emphasizes a model's understanding of **shape diversity**. A high score suggests the model can robustly classify sequences with complex arrangements of different shapes, a key aspect of the task.

#### **Metric 2: Color-Weighted Accuracy (CWA)**

* **Description**: This metric heavily prioritizes performance on instances with high color variety.  
* $\text{CWA} = \frac{\sum_{i=1}^{N} w_i^{\text{CWA}} \cdot \mathbb{I}(y_i = \hat{y}_i)}{\sum_{i=1}^{N} w_i^{\text{CWA}}}, \quad \text{where } w_i^{\text{CWA}} = C_c(S_i)$ 
* **Rationale (as presented to the AI agent)**: CWA emphasizes a model's understanding of **color diversity**. A high score suggests the model can robustly  sequence complex combinations of different colors, a key aspect of the task.

### SOTA Performance 

Below are the SPR_BENCH benchmark along with its corresponding State-of-the-Art (SOTA) performance on SWA and CWA metric. The SOTA represent the best performance achieved on the SPR_BENCH benchmark and provide a reference point for evaluating the effectiveness of the models trained on these datasets.

| Name          | Metric                            | SOTA Performance |
| ------------- | --------------------------------- | ---------------- |
| **SPR_BENCH** | **SWA** (Shape-Weighted Accuracy) | 70.0 %           |
| **SPR_BENCH** | **CWA** (Color-Weighted Accuracy) | 65.0 %           |


### **Data Description**

```
Directory layout expected:
SPR_BENCH/
 ├─ train.csv   (20,000 rows)
 ├─ dev.csv     (5,000 rows)
 └─ test.csv    (10,000 rows)

```

Each `*.csv` has **UTF-8**, comma-separated columns:

| id           | sequence                 | label |
| ------------ | ------------------------ | ----- |
| SPR_train_17 | ▲r ■b ▲r ●g ◆r ■r ●y ◆r | 1     |

* A **token** is a shape glyph plus an optional 1-letter colour (`▲g`, `■`, …).
* **Label** = `1` (accept) or `0` (reject).


## Task Instructions

1. **Design an Algorithm**: Develop an algorithm to solve the **SPR (Symbolic Pattern Recognition)** task. Your algorithm should decide whether a given \$L\$-token sequence of abstract symbols satisfies the hidden target rule.

2. **Training Procedure**:
   * Train your model using the **Train split** of each selected benchmark.
   * Tune your model on the **Val split**.
   * The **Test split** labels are withheld, and you must report your model’s performance on this unseen data.

3. **Baseline and Metrics**:
   * Use evaluation metrics like Shape-Weighted Accuracy (SWA) and Color-Weighted Accuracy (CWA) for performance comparison.
   * Use the current State-of-the-Art (SOTA) performance on the SPR_BENCH as the baseline.

4. **Submission Requirements**: For SPR_BENCH, submit a **separate model** along with the following:
   * The final performance on the Test set of the **selected metrics**.
   * A detailed comparison of your model’s performance against the SOTA performance on your chosen metrics.

5. **Objective**: The goal of this task is to develop a **robust algorithm** that:
   * Demonstrates strong generalization across variations in **vocabulary sizes**, **sequence lengths**, and **rule complexities**.