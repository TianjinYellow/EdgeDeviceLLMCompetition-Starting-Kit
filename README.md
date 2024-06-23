# Starting Kit for Edge-Device LLM Competition 2024 

This is the starting kit for the Edge-Device LLM Competition, a NeurIPS 2024 competition. To learn more about the competition, please see the [competition website](https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/).  This starting kit provides instructions on downloading data, running evaluations, and generating submissions.

# Submission Requirements

- Model definition file (.py) and its configuration file: Must be huggingface format. (https://huggingface.co/docs/transformers/en/custom_models)

- The saved weights: Must be huggingface format, i.e., saved via save_pretrained() function that inherated from transformers.PreTrainedModel(https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/model#transformers.PreTrainedModel)

- The compiled model via MLC-MiniCPM tool. (https://github.com/OpenBMB/mlc-MiniCPM)

- The evaluated results including sores for CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA + Throughput+Memory Usage

# Evalution for CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA Tasks

Evaluation for CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA Tasks is based on Opencompass tool. 

- Environment Setup

```bash
  conda create --name opencompass python=3.10 
  conda activate opencompass
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  cd opencompass && pip install -e .
  cd opencompass/human-eval && pip install -e .
```

- Data Preparation

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

- Evaluation 
  - Evaluate Huggingface models.

```bash
# --dataset 
python run.py --datasets ceval_ppl mmlu_ppl --hf-type base --hf-path huggyllama/llama-7b
```
