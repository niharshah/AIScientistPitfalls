# AIScientistPitfalls

## Overview

This repository accompanies the paper *"Understanding and Mitigating Pitfalls in AI-Driven Scientific Research"* and aims to provide a structured framework for detecting and mitigating common failure modes in AI-driven scientific discovery. As AI systems increasingly take on roles traditionally held by human researchers—such as hypothesis generation, experimentation, and interpretation—it becomes critical to understand where and how these systems can go wrong.

## Repository Structure

```
AIScientistPitfalls/
├── AI Scientist v2/       # Case studies of The AI Scientist v2
│   ├─ generated_research  # including the generated research under four pitfall detection experiments
│   └─...
├── AgentLaboratory/       # Case studies of Agent Laboratory
│   ├─ generated_research  # including the generated research under four pitfall detection experiments
│   └─...
├── SPR Task/              # incuding datasets and task description for four identified pitfalls
│
├── pitfall_detection/     # including datasets and code for pitfall detection method 
│                            
└─ README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API or other LLM access

### Installation

```bash
git clone https://github.com/niharshah/AIScientistPitfalls.git
cd AIScientistPitfalls
```

### Usage

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


## Citation

If you use this repository in your research, please cite:

```

```


