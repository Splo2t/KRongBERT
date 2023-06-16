# KRongBERT
The bidirectional encoder representations from transformers (BERT) model has achieved remarkable success in various natural language processing tasks for languages based on the Latin alphabet. However, the Korean language, characterized by limited data resources and intricate linguistic structures, presents substantial challenges and constraints. In this paper, we introduce KRongBERT, a morphological approach tailored to effectively understand and analyze the unique features of the Korean language. The KRongBERT approach mitigates the out-of-vocabulary issues that arise with byte-pair-encoding tokenizers in Korean and provides language-specific embedding layers to support in-depth understanding. The experiments reveal that KRongBERT outperforms the existing Korean text encoders. 

## Repository Structure
**KRongBERT_venv** This is a Python virtual environment for running KRongBERT. The Transformers library has been modified to implement KRongBERT, and the modified library is installed in the virtual environment. The small_data.csv inside the folder is part of the actual dataset used by KRongBERT.

**pre-train** It contains code for pre-training KRongBERT. The code for pre-training does not have the Affix-aware Korean Tokenization technique applied inside krong_tokenizer.py using the pre-encoded dataset due to implementation issues. Please refer to the finetune code for that part.

**finetune** It contains code to fine-tune KRongBERT to perform Korean downstream tasks.

## Benchmark Score
| Models            | Data | NSMC  | NER   | KorNLI | KorSTS |
| ----------------- | ---- | ----- | ----- | ------ | ------ |
| Multilingual BERT | 2GB  | 87.5  | 80.3  | 76.8   | 77.8   |
| HanBERT           | 70GB | 90.16 | 87.31 | 80.89  | 83.33  |
| KoBERT            | 20GB | 90.1  | 86.11 | 79     | 79.64  |
| KcBERT-base       | 15GB | 89.62 | 84.34 | 74.85  | 75.57  |
| KoreALBERT-base   | 43GB | 89.6  | 82.3  | 79.7   | 81.2   |
| KoreALBERT-Large  | 43GB | 89.7  | 83.7  | 81.1   | 82.1   |
| KRongBERT         | 30GB | 90.22 | 86.79 | 80.92  | 84.6   |
