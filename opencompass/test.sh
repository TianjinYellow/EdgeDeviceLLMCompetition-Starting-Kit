#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=1-20:00:00
#SBATCH --output=out.out

#source activate EdgeLLMCompetition
#cd opencompass
CUDA_VISIBLE_DEVICES=0 python run.py --datasets commonseqa_gen longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen crowspairs_gen --hf-num-gpus 1 --hf-type base --hf-path microsoft/phi-2 --debug --model-kwargs device_map='auto' trust_remote_code=True

#tested-model
#microsoft/phi-2
#truthfulqa_gen
#tested-datasets
#commonseqa_gen+longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen

#/scratch-shared/HTJ/llama_prune/pytorch_model.bin
