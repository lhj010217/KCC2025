# Automated Few-Shot Tabular Learning Framework via LLM-Based Semantic Feature Engineering

<p align="center">
<img width="2934" height="1380" alt="Component 5 (2)" src="https://github.com/user-attachments/assets/22fd0c50-cae2-4648-8185-893e74c8a359" />

</p>

This repository contains the official implementation of the paper: **[Automated Few-Shot Tabular Learning Framework via LLM-Based Semantic Feature Engineering](#)**.
https://github.com/lhj010217/KCC2025/blob/main/README.md
Our framework leverages Large Language Models (LLMs) and attention mechanisms to automate semantic feature engineering for tabular data, specifically designed for few-shot learning environments where labeled data is extremely scarce. It combines LLM-generated features with a neural network-based feature-bagging ensemble to achieve superior performance over traditional AutoML solutions.

## Prerequisites

### 1. Local LLM Setup (Ollama)
This framework uses a local instance of [Ollama](https://ollama.ai/) to generate features securely without sending sensitive tabular data to external APIs.
- Install Ollama.
- Pull the Llama 3.1 model:
  ```bash
  ollama run llama3.1
  ```
- Ensure the Ollama server is running locally on port `11434`.

## Usage

### 1. Prepare Datasets
Place your datasets in the `datasets/` directory. The framework expects a structure like:
```text
datasets/
  ├── bone tumor/
  │   └── bone tumor.csv
  ├── Heart Attack/
  │   └── Heart Attack.csv
  └── ...
```

### 2. Run the Proposed Framework
Execute the main training script to perform LLM-based feature engineering and train the Feature-Bagging Neural Network model.
```bash
python main.py
```
*(Alternatively, you can run `exp.py` for ablation studies and detailed split experiments.)*

### 3. Run AutoML Baselines
To compare the proposed framework with existing AutoML solutions (TPOT, AutoGluon, MLJAR, H2O, TabPFN), run:
```bash
python AutoML.py
```

## Requirements

The core framework requires the following dependencies:
- `torch`
- `tensorflow`
- `transformers`
- `pandas`
- `scikit-learn`
- `requests`

For baseline comparisons (`AutoML.py`), you will additionally need:
- `h2o`
- `tpot`
- `autogluon`
- `lightautoml`
- `mljar-supervised`
- `tabpfn`

Install core dependencies using:
```bash
pip install -r requirements.txt
```

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{lee2026automated,
  title={Automated Few-Shot Tabular Learning Framework via LLM-Based Semantic Feature Engineering},
  author={Lee, Hyungjoon and Kim, Juntae},
  journal={TBD},
  year={2026}
}
```
