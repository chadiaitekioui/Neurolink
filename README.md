# Neurolink

Modular pipeline to forecast emergent neuroscience research directions from PubMed literature.

**Literature LoRA**: fine-tune Mistral-7B on temporal question pairs, with benchmark against Mistral-7B base and BrainGPT.

image

## Why these models?

Neurolink compares three 7B-scale causal LMs on the same forecast task: propose novel neuroscience research questions for a target year, given impact-ranked context from prior years. The trio isolates three hypotheses:


| Model             | Hypothesis tested                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `mistral_base`    | A general-purpose LLM already captures temporal research trends without domain fine-tuning |
| `literature_lora` | Temporal LoRA on indexed question pairs teaches the model to forecast what comes next      |
| `braingpt`        | A neuroscience-specialized pre-training (BrainGPT) outperforms raw Mistral                 |




### Capabilities

All three models share identical generation prompts and inference settings (`temperature: 0`, greedy decoding) so differences reflect model knowledge, not prompt engineering.

- **Mistral-7B-v0.1** (`mistral_base`): open general-purpose baseline; no adapter, no domain bias. Serves as the null hypothesis: can a stock LLM forecast emergent neuroscience directions from context alone?
- **Literature LoRA** (`literature_lora`): Mistral-7B + a LoRA adapter trained on temporal pairs (context ≤ year *T* → questions at *T+1*). The adapter is frozen at anchor `year_max` to simulate real forecast conditions where the model cannot see future literature.
- **BrainGPT-7B-v0.2** (`braingpt`): domain LLM pre-trained on neuroscience text ([BrainGPT](https://huggingface.co/BrainGPT/BrainGPT-7B-v0.2)). Tests whether broad neuro-domain knowledge transfers better than corpus-specific temporal fine-tuning.

At inference, each model receives up to 40 high-impact questions as context, generates *k* novel questions (20–200 words, must end with `?`), and ranks candidates by completion likelihood. GPU + 4-bit quantization required for training and benchmark stages.

Full protocol (index → dual LoRA anchors → benchmark → eval → forecast): see **[Protocol.md](Protocol.md)**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[train]"   # LoRA + 4-bit (GPU)
```

Set `HF_TOKEN` for gated models (Mistral-7B, BrainGPT).

## Quick start

```bash
python -m neurolink menu
```

Menu: **Index → LoRA → Benchmark → Complete workflow (GPU) → Status**.

image

Or run the full protocol in one command (GPU required):

```bash
neurolink workflow
neurolink workflow --skip-index   # if index already built
```

## CLI reference


| Command                                   | Role                          |
| ----------------------------------------- | ----------------------------- |
| `neurolink index`                         | Full index pipeline           |
| `neurolink train-literature --year-max T` | Train LoRA through year T     |
| `neurolink compare --lora-year-max T`     | Benchmark 3 LLMs on years > T |
| `neurolink eval --config …`               | Metrics on latest predict run |
| `neurolink workflow`                      | Complete protocol (above)     |
| `neurolink menu`                          | Interactive UI                |


## Reference

Pipeline config inspired by [MMORE RAG Pipeline](https://arxiv.org/html/2509.11937v1) (Light Laboratory, Swiss-AI @EPFL).

Motivated by [Large language models surpass human experts in predicting neuroscience results](https://www.nature.com/articles/s41562-024-02046-9).