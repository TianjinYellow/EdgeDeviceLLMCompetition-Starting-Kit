from opencompass.models import CustomizeLlama

# Please follow the instruction in the Meta AI website https://github.com/facebookresearch/llama/tree/llama_v1
# and download the LLaMA model and tokenizer to the path './models/llama/'.
#
# The LLaMA requirement is also needed to be installed.
# *Note* that the LLaMA-2 branch is fully compatible with LLAMA-1, and the LLaMA-2 branch is used here.
#
# git clone https://github.com/facebookresearch/llama.git
# cd llama
# pip install -e .

models = [
    dict(
        abbr='example',
        type=CustomizeLlama,
        path='/scratch-shared/HTJ/model_llama',
        tokenizer_path='/scratch-shared/HTJ/model_llama/tokenizer.model',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
