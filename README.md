# AIScientistPitfalls

## Overview

This repository accompanies the paper [The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems](https://arxiv.org/abs/2509.08713) and aims to provide a empirical framework for detecting common failure modes in AI-driven scientific discovery, including inappropriate benchmark selection, metric misuse, data leakage, post-hoc selection bias. As AI systems increasingly take on roles traditionally held by human researchers such as hypothesis generation, experimentation, and paper writing, it becomes critical to understand where and how these systems can go wrong.

## Repository Structure

```
AIScientistPitfalls/ v2
├── AI Scientist v2/       # Case studies & experiments with The AI Scientist v2
│   ├─ generated_research  # Generated research under four pitfall detection experiments
│   └─...                  # Other supporting code / configs
├── AgentLaboratory/       # Case studies & experiments with Agent Laboratory
│   ├─ generated_research  # Generated research under four pitfall detection experiments
│   └─...
├── SPR Task/              # Contains datasets and task descriptions for four identified pitfalls
│
├── pitfall_detection/     # Contains datasets and code for pitfall detection method 
│                            
└─ README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Access to large language model(s) (e.g. via OpenAI API)

### Installation

```bash
git clone https://github.com/niharshah/AIScientistPitfalls.git
cd AIScientistPitfalls
```

### Using the **SPR Task / Pitfall Detection** Module
* The SPR Task is a fully synthetic task outside the scope of internet-scale corpora, allowing you to detect four failure modes.
* Under `SPR Task/` are task descriptions (`SPR.md`), datasets (`SPE_BENCH/`) and code for loading the datasets (`SPR.py`) for the four identified pitfalls.

* You can use these to evaluate whether your AI scientist system exhibits any of these four types of pitfalls.

* Typical usage:

  1. Put the task descriptions, datasets and code at the right place under your AI scientist system workspace.
  2. Run your system and generate research under the detection tasks.
  3. Inspect results according to the failure modes flagged in our paper.


---

### Reseach generation

To generate a paper with The AI Scientist v2:

```bash
cd AI Scientist v2
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
bash run.sh
```
This should generate the research outputs (project) under a new created folder `experiments`.

To generate a paper with AgentLaboratory:

```bash
cd AgentLaboratory
# Set up and Activate Python Environment
python -m venv venv_agent_lab
# Now activate this environment:
source venv_agent_lab/bin/activate
# Install pdflatex [OPTIONAL]
sudo apt install pdflatex
# Install required libraries
pip install -r requirements.txt
bash run.sh
```

This similarly produces the generated research under a new created folder `MATH_research_dir`.


## Pitfall Detection (`pitfall_detection/`)
The pitfall_detection module provides tools to identify four methodological pitfalls in AI-generated scientific research outputs. 

### Usage

1. Go to the directory:

   ```bash
   cd pitfall_detection
   ```
2. Prepare your the following AI-generated research outputs in correct paths:
      - Task description
            
            For the inappropiate benchmarch selection bias issue, the list of available benchmarks can be retrieved from the internet by LLMs based on the task relevance.
      - Generated paper
      - Code

            For the post-hoc selection bias issue, the code and logs should include all the experiments attempted by the AI scientist system.
      - Logs 

3. Run the detection pipeline:

   ```bash
   python detect_pitfalls.py
   ```
4. Inspect the output for detected pitfalls and their descriptions.


## Citation

If you use this repository in your research, please cite:

```
@misc{luo2025automateseehiddenpitfalls,
      title={The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems}, 
      author={Ziming Luo and Atoosa Kasirzadeh and Nihar B. Shah},
      year={2025},
      eprint={2509.08713},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.08713}, 
}
```


