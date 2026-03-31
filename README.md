# Human vs Machine Translation Detection (Cross-Model/Cross-Domain/Low-Resource)

This repo supports reproducible experiments to detect machine-generated translation vs human translation.  
It includes data preparation, model fine-tuning, inference generation, and multi-metric quality analysis across:
- languages: Afrikaans (af), English (en), Spanish (es), Swahili (sw)
- datasets: FLORES, EU DGT, OPENWHO, SMOL
- models: LLM and dedicated translation-detection models
- data split: train/val/test and cross-domain evaluation sets  

---

## 📁 Root files

- README.md (original brief project description)
- Data (experiment datasets and evaluation results)
- `Dataset Generation/` (script notebooks for building data)
- Fine-tuning (notebooks for training translation/model detector)
- Inference (runtime generation and detection evaluation notebooks)
- `Quality Evaluation/` (analysis notebooks with metrics and correlations)

---

## 📂 Data

### Structure
- Data
  - `eval/` (evaluation-ready test outputs, predictions, model comparators)
  - `train/` (train/val csvs split by language)

### `eval` subsets
- `eu_dgt_csvs/`  
  - `en/`, `es/`  
  - files:
    - `eu_dgt_output_with_models.csv` etc: model outputs merged with ref/human
    - `eu_dgt_test_{model}.txt`: baseline output of each MT model
    - `eu_dgt_model_mdeberta_sent_col_human translation.csv`: reference human translation
- `flores/`  
  - `af/`, `sw/`  
  - same pattern: `flores_output_with_models*.csv`, `flores_test_{model}.txt`, `flores_test.csv`.
- `openwho_csvs/` and `smol_csvs/` (similar)
  
### `train`
- `train/af/train.csv`, `train/af/val.csv`
- `train/en/train.csv`, `train/en/val.csv`
- `train/es/train.csv`, `train/es/val.csv`
- `train/sw/train.csv`, `train/sw/val.csv`

Purpose:
- training detector/classifier on bilingual pairs or feature engineered data
- validation for tuning accuracy
- test for final evaluation in eval

---

## 🧩 Dataset Generation

- `Dataset Generation/Data Preprocessing/` notebooks:
  - `build_eu_dgt_pal.ipynb`
  - `build_swahili_flores_plus.ipynb`
  - `build_test_out_domain_openwho.ipynb`
  - `build_test_out_domain_smol_sw_af.ipynb`
  - `build_test_out_domain_smol.ipynb`
  - `chunk_combined_dataset.ipynb`
  - `combine_eu_dgt_tranlsated.ipynb`
- `Dataset Generation/Data Tranlslation/` notebooks:
  - `api_based_translations.ipynb`
  - `loaded_model_translations.ipynb`

Purpose:
- convert raw sources into labeled detection datasets
- perform chunking and combining for cross-model training
- produce machine translation outputs via API (OpenAI/Gemini/Claude/ etc.)
- include offline translation by locally loaded models for reproducibility

---

## 🧠 Fine-tuning

- fine_tune_afrikans.ipynb
- fine_tune_english.ipynb
- fine_tune_spanish.ipynb
- fine_tune_swahili.ipynb

Purpose:
- training or fine-tuning classification/detection model for each language
- includes dataset read, tokenizer, train loop, eval metrics
- may save checkpoints and final model artifacts in cell outputs or external paths

---

## 🧪 Inference

- Decoder.ipynb
- Encoder_sent.ipynb

Purpose:
- inference pipeline for models
- decoder-based generation for translation detection or alt scoring
- encoder sentence-level features and detection score generation
- probably produces output files in eval

---

## 📊 Quality Evaluation

- `Quality Evaluation/analyze_quality.ipynb`
- `Quality Evaluation/calculate_score.ipynb`
- `Quality Evaluation/correlation.ipynb`

Purpose:
- compute quality metrics (BLEU, COMET, BERTScore)
- analyze translation quality vs detection performance
- correlation analysis (e.g., quality vs classification confidences / human-likeness)
- produce tables and plots used in paper

---
