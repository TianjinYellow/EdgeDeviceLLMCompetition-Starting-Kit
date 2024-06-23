#!/bin/bash
cd opencompass
CUDA_VISIBLE_DEVICES=0 python run.py --datasets humaneval_gen --hf-num-gpus 1 --hf-type base --models example --debug --model-kwargs device_map='auto' trust_remote_code=True

#tested-model
#microsoft/phi-2

#tested-datasets
#commonseqa_gen+longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen truthfulqa_gen

#/scratch-shared/HTJ/llama_prune/pytorch_model.bin
