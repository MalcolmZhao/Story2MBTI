# Story2MBTI

This repository contains scripts and data related to the analysis and evaluation of BERT models for MBTI classification as the group project for Berkeley INDENG 242B Group: Story to MBTI

## Structure

- **data**: Contains the data used for training and evaluation.
  - `MVMR_BERT_tok.pkl`: Source file used by `1.data_extract.ipynb` for data extraction and preprocessing.
  - `MVMR_BERT_tok`: Processed data resulting from `1.data_extract.ipynb`.
- **results**: Stores the results and logs generated during model training and evaluation.
  - `20240422101530_log.pkl`: Training logs for `2.analysis_with_transformer.py`.
  - `20240429213105_log.pkl`: Training logs for `3.analysis_with_bert.py`.
- `1.data_extract.ipynb`: Jupyter Notebook for data extraction and preprocessing.
- `2.analysis_with_transformer.py`: Python script for analysis using a transformer-based model.
- `3.analysis_with_bert.py`: Python script for analysis using a BERT model.
- `4.bert_model_evaluation.py`: Python script for evaluating the BERT model using the `mbti_bert_model.py` class.
- `mbti_bert_model.py`: Final BERT model class used for MBTI classification.

Please note that the final model weights are not included in the repository due to their size exceeding the limit for GitHub. The model weight file is approximately 1.28 GB.

## Usage

1. **Data Extraction and Preprocessing**: Execute `1.data_extract.ipynb` to preprocess the data.
2. **Analysis with Transformer Model**: Run `2.analysis_with_transformer.py` to perform analysis using a transformer-based model.
3. **Analysis with BERT Model**: Run `3.analysis_with_bert.py` to perform analysis using a BERT model.
4. **Model Evaluation**: Execute `4.bert_model_evaluation.py` to evaluate the BERT model using the `mbti_bert_model.py` class.

## Dependencies

- Python 3.12
- PyTorch
- Transformers library
- Other dependencies as listed in individual scripts


