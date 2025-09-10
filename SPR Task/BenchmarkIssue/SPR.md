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

## 20 Benchmarks from HuggingFace

In this experiment, we use 20 carefully curated benchmarks sourced from HuggingFace, each designed to evaluate the performance of machine learning models on a symbolic pattern recognition task. These benchmarks are unified in terms of dataset splitting, label distribution, and evaluation metrics, ensuring consistency across experiments. The benchmarks include a variety of symbolic patterns that challenge models to classify sequences of abstract symbols under varying conditions.


### **Fixed Global Dataset Parameters**

All benchmarks are standardized with the following fixed global parameters:


| Field                         | Value                                |
| ----------------------------- | ------------------------------------ |
| **Instances per split**       | Train = 2,000 Dev = 500 Test = 1,000 |
| **Label balance**             | 50 % accept / 50 % reject            |
| **Public leaderboard metric** | Label Accuracy on each benchmark     |

### Benchmark Results and SOTA Accuracy

Below are the names of the 20 benchmarks (encrypted for privacy) along with their corresponding State-of-the-Art (SOTA) accuracy scores. These accuracy values represent the best performance achieved on each benchmark and provide a reference point for evaluating the effectiveness of the models trained on these datasets.

| Name      | SOTA Accuracy |
| --------- | ------------- |
| **FWZGE** | 68.9 %        |
| **ROMNH** | 62.9 %        |
| **SFRFG** | 55.1 %        |
| **EWERV** | 66.4 %        |
| **IJSJF** | 60.8 %        |
| **IDWEP** | 58.7 %        |
| **TSHUY** | 54.7 %        |
| **GURSG** | 52.3 %        |
| **QAVBE** | 71.3 %        |
| **DFWZN** | 60.6 %        |
| **LYGES** | 72.6 %        |
| **PHRTV** | 53.6 %        |
| **PWCGE** | 57.2 %        |
| **ZAEFE** | 56.9 %        |
| **TEZGR** | 69.6 %        |
| **IRXBF** | 70.4 %        |
| **URCJF** | 61.4 %        |
| **JWAEU** | 63.5 %        |
| **MNSDE** | 65.5 %        |
| **TEXHE** | 58.3 %        |

The benchmarks are encrypted to preserve privacy, but the results indicate the variability in performance across different benchmarks, with accuracy ranging from 52.3% to 72.6%. These benchmarks were designed to evaluate the robustness and generalization capabilities of models in detecting hidden patterns in symbolic sequences, making them a valuable resource for advancing machine learning techniques for symbolic reasoning.


### **Data Description**

```sql
SPR_BENCH/
 ├─ SFRFG/
 │   ├─ train.csv   (2,000 rows)
 │   ├─ dev.csv     (  500 rows)
 │   └─ test.csv    (1,000 rows)
 ├─ IJSJF/
 └─ …

```

Each `*.csv` has **UTF-8**, comma-separated columns:

| id           | sequence                | label |
| ------------ | ----------------------- | ----- |
| B01_train_17 | ▲r ■b ▲r ●g ◆r ■r ●y ◆r | 1     |

* A **token** is a shape glyph plus an optional 1-letter colour (`▲g`, `■`, …).
* **Label** = `1` (accept) or `0` (reject).


## Task Instructions

1. **Design an Algorithm**: Develop an algorithm to solve the **SPR (Symbolic Pattern Recognition)** task. Your algorithm should decide whether a given \$L\$-token sequence of abstract symbols satisfies the hidden target rule.

2. **Benchmark Selection**: From the 20 available benchmarks listed in the following section, **select 4 benchmarks** to evaluate your algorithm. Provide a **justification** for your choice of benchmarks based on their characteristics and how they align with your algorithm’s strengths.

3. **Training Procedure**:
   * Train your model using the **Train split** of each selected benchmark.
   * Tune your model on the **Dev split**.
   * The **Test split** labels are withheld, and you must report accuracy based on your model’s performance on this unseen data.
   * Note that **cross-benchmark training** is prohibited. Each model should be trained and evaluated independently for each chosen benchmark.

4. **Baseline Comparison**: Set the **SOTA (State-of-the-Art) accuracies** for each benchmark as a baseline. Your goal is to **compare your model’s performance** against these baselines and demonstrate improvements.

5. **Submission Requirements**: For each selected benchmark, submit a **separate model** along with the following:
   * The **final accuracy on the Test set**.
   * A comparison of your model’s performance against the **SOTA baseline** for that benchmark.

6. **Objective**: The goal of this task is to develop a **robust algorithm** that:
   * Demonstrates strong generalization across variations in **vocabulary sizes**, **sequence lengths**, and **rule complexities**.


