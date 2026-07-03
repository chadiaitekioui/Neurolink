# Neurolink

Modular pipeline to forecast emergent neuroscience research directions from PubMed literature.

**Literature LoRA**: fine-tune Mistral-7B on temporal question pairs, with optional benchmark against Mistral-7B base and BrainGPT.

![image](neurolink.png)

## Installation

**Venv**

```bash
cd neurolink
python -m venv .venv
source .venv/bin/activate
```

**Dependencies**

```bash
pip install -e .
# optional: LoRA training
# pip install -e ".[train]"
```

**CLI / menu**

```bash
python -m neurolink menu
```

Menu: **Index → LoRA → Benchmark → Status**.

## Literature LoRA

- `train-literature` : fine-tune LoRA on pairs context ≤ T → questions at T+1.
- `predict` / `literature` : prompt with top questions ≤ N−1 → generate novel questions for N.
- `compare` : benchmark `literature_lora`, `mistral_base`, `braingpt` on years after a saved LoRA `year_max`.
- Code: `forecast/predict/literature_lora.py`, `forecast/train.py`, `forecast/benchmark.py`.

The LoRA adapter is saved locally under `data/models/literature/year_max_*/lora/`.
Inference can run **without retraining** when the adapter for `year_max_{N-1}` exists.

```bash
# Train once
neurolink train-literature --year-max 2023 --config config/forecast/predict_literature.yaml

# Infer only (no train_literature stage)
neurolink literature --config config/forecast/pipeline_literature.yaml
```

Train and predict in one command: `config/forecast/pipeline_literature_train.yaml`.

## Evaluation

TF-IDF semantic matching between generated predictions and ground-truth questions for year N.

Metrics: precision@k, recall@k, BrainBench-style perplexity discrimination, LoRA contamination audit.

## Reference

Pipeline config files inspired by
[MMORE RAG Pipeline](https://arxiv.org/html/2509.11937v1) (Light Laboratory, Swiss-AI @EPFL).

Motivated by [Forecasting emerging research directions](https://www.nature.com/articles/s41562-024-02046-9).
