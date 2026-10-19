# Human vs Machine Translation Detection

Code and data for the COLM 2026 paper **Human vs Machine Translation Detection: A Cross-Model, Cross-Domain, and Low-Resource Analysis**.

**Paper:** [OpenReview](https://openreview.net/forum?id=Jqx5zQM5IQ)
**Authors:** Yabsera Yemanberhan, Prasenjit Mitra (Carnegie Mellon University Africa), Roald Eiselen (North-West University)

## Overview

Web-crawled training corpora often contain machine-translated (MT) text presented as human translation (HT). This repository studies how well MT can be told apart from HT across:

| Dimension | Coverage |
|---|---|
| Languages | English, Spanish (high-resource); Swahili, Afrikaans (low-resource) |
| Domains | EU DGT (legislative), OpenWHO (medical), SMOL (conversational), FLORES+ (benchmark) |
| MT systems seen in training | Qwen2.5-3B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-IT, NLLB-200-3.3B |
| MT systems unseen in training | GPT-5, Claude Opus 4, Gemini 2.5 Pro |
| Detectors | Fine-tuned mDeBERTa-v3-base and XLM-RoBERTa-base (sentence and 300-token chunk inputs); zero-shot LLMs (GPT-5, Claude Opus 4, Gemini 2.5 Pro) |
| Quality metrics | COMET (`Unbabel/wmt22-comet-da`), BLEURT-20, BLEU |

Main findings: LLMs recognise human translation well but rarely flag fluent MT. Fine-tuned encoders are strong in domain but lose MT recall on unseen translators and domains. Training and evaluating on 300-token chunks substantially improves generalisation. Lower-quality MT is easier to detect.

## Repository layout

```
Data/
  train/{en,es,sw,af}/{train,val}.csv     Labelled training and validation splits
  eval/
    eu_dgt_csvs/{en,es}/                  Legislative test sets
    openwho_csvs/{en,es}/                 Medical test sets (out of domain)
    smol_csvs/{en,sw,af}/                 Conversational test sets (out of domain)
    flores/{sw,af}/                       Low-resource benchmark test sets
Dataset Generation/
  Data Preprocessing/                     Build HT pools, test sets, and chunked inputs
  Data Tranlslation/                      Generate MT with local models and APIs
Fine-tuning/                              One notebook per language for encoder training
Inference/                                Encoder and LLM detection over test sets
Quality Evaluation/                       COMET/BLEURT scoring and quality vs detectability analysis
```

### Data formats

**Training splits** (`Data/train/<lang>/train.csv`, `val.csv`): two columns, `sentence` and `label`, where **0 = HT** and **1 = MT**. Classes are balanced.

| Language | Train | Val | Source pool |
|---|---|---|---|
| English | 110,000 | 10,000 | EU DGT |
| Spanish | 110,000 | 10,000 | EU DGT |
| Swahili | 3,000 | 500 | FLORES+ |
| Afrikaans | 3,000 | 500 | FLORES+ |

**Test sets** (`Data/eval/...`):

| File pattern | Contents |
|---|---|
| `*_test_<system>.txt` | Raw output of one MT system, one sentence per line |
| `*_output_with_models.csv` | Sentence-level test set: `human translation`, `source text`, `source language`, plus one column per MT system (`claude`, `gemini`, `gemma`, `llamma`, `nllb`, `openai`, `qwen`) |
| `*_mdeb_chunk*.csv`, `*_xlmr_chunk*.csv` | The same test set merged into chunks of up to 300 tokens, using the mDeBERTa or XLM-R tokenizer |

## Setup

Python 3.10 or later and a CUDA GPU are recommended. The paper's experiments ran on a workstation with three NVIDIA RTX A5000 GPUs (24 GB each) and CUDA 12.2.

```bash
git clone https://github.com/Yabsera-Haile/Human-vs-Machine-Translation-Detection-Codebase.git
cd Human-vs-Machine-Translation-Detection-Codebase
pip install -r requirements.txt
# BLEURT is only needed for Quality Evaluation
pip install git+https://github.com/google-research/bleurt.git
```

API-based steps (commercial translation and LLM detection) use an OpenAI-compatible client. Set your key before running them:

```bash
export API_KEY=...
```

Each notebook has its paths and settings in the first cells (for example `INPUT_DIR`, `MODEL_NAME`, `INPUT_CSV`). Fill these in before running.

## Reproducing the experiments

The released `Data/` folder already contains the training splits and test sets, so you can start at step 3. Steps 1 and 2 rebuild the data from the original sources.

### 1. Build human translation pools and test sets

| Notebook | What it does |
|---|---|
| `build_eu_dgt_pal.ipynb` | Parses the EU DGT translation memory (TMX) into aligned source/HT pairs |
| `build_swahili_flores_plus.ipynb` | Builds English-Swahili and English-Afrikaans parallel sets from `openlanguagedata/flores_plus` |
| `build_test_out_domain_openwho.ipynb` | Builds the medical test set from `raphaelmerx/openwho` |
| `build_test_out_domain_smol.ipynb`, `build_test_out_domain_smol_sw_af.ipynb` | Build conversational test sets from `google/smol` |

### 2. Generate machine translations and labelled splits

| Notebook | What it does |
|---|---|
| `loaded_model_translations.ipynb` | Translates source text with local open models via `transformers` (set `model_name`) |
| `api_based_translations.ipynb` | Translates source text with commercial models through an OpenAI-compatible API, temperature 0 |
| `combine_eu_dgt_tranlsated.ipynb` | Combines MT outputs and HT into balanced, labelled train/val/test splits |
| `chunk_combined_dataset.ipynb` | Merges consecutive test rows into chunks of up to 300 tokens for chunk-level evaluation |

### 3. Fine-tune the encoder detectors

Open `Fine-tuning/fine_tune_<language>.ipynb` and set:

- `MODEL_NAME`: `microsoft/mdeberta-v3-base` or `xlm-roberta-base`
- `INPUT_DIR`: for example `Data/train/en`
- `chunked`: `True` for 300-token chunks, `False` for sentences

Training uses the Hugging Face `Trainer` with AdamW (weight decay 0.01), linear warmup and decay, fp16, and per-epoch evaluation. The checkpoint with the best validation F1 is kept. Chunked splits are shuffled with seed 42. Per-language settings from the paper:

| Lang | Input | Epochs | LR | Batch | Grad. accum. | Warmup |
|---|---|---|---|---|---|---|
| EN | sentence | 3 | 2e-5 | 8 | 4 | 0.06 |
| EN | chunk | 4 | 2e-5 | 8 | 2 | 0.05 |
| ES | sentence | 3 | 2e-5 | 4 | 4 | 0.06 |
| ES | chunk | 4 | 2e-5 | 4 | 4 | 0.05 |
| SW | sentence | 5 | 1.5e-5 | 8 | 1 | 0.1 |
| SW | chunk | 6 | 1e-5 | 8 | 1 | 0.1 |
| AF | sentence | 6 | 1e-5 | 8 | 1 | 0.1 |
| AF | chunk | 6 | 1e-5 | 8 | 1 | 0.1 |

The final model and tokenizer are saved to `<OUTPUT_DIR>/final_model`.

### 4. Run detection

| Notebook | What it does |
|---|---|
| `Inference/Encoder_sent.ipynb` | Runs a fine-tuned checkpoint over every translation column of a test CSV (batch size 16, max length 512) and appends the predicted label and HT/MT probabilities |
| `Inference/Decoder.ipynb` | Zero-shot LLM detection. Each text is classified 3 times at temperature 0; the final label is the majority vote and the confidences are averaged |

Set `MODEL_NAME` in `Decoder.ipynb` to `gpt-5`, `gemini-2.5-pro`, or your provider's Claude Opus 4 model ID. The few-shot prompts are given in Appendix A.1 of the paper.

### 5. Quality vs detectability

| Notebook | What it does |
|---|---|
| `calculate_score.ipynb` | Scores every MT column against the human reference with COMET, BLEURT-20, and BLEU |
| `correlation.ipynb` | Pearson and Spearman correlations between quality scores and detector confidence |
| `analyze_quality.ipynb` | Summary tables and plots |

## Citation

```bibtex
@inproceedings{yemanberhan2026human,
  title     = {Human vs Machine Translation Detection: A Cross-Model, Cross-Domain, and Low-Resource Analysis},
  author    = {Yemanberhan, Yabsera and Mitra, Prasenjit and Eiselen, Roald},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Jqx5zQM5IQ}
}
```

## Data sources and licences

This repository redistributes text derived from EU DGT, FLORES+, OpenWHO, and SMOL. Please follow the original licence of each dataset when reusing the data.
