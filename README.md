<h2 align="center">Starting Kit for Edge-Device LLM Competition, NeurIPS 2024</h2>

This is the starting kit for the Edge-Device LLM Competition, a NeurIPS 2024 competition. To learn more about the competition, please see the [competition website](https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/).  This starting kit provides instructions on downloading data, running evaluations, and generating submissions.

<span style="color:red"><strong>Please join us on Discord for discussions and up-to-date announcements:</strong></span>

[https://discord.gg/SsyY2s2k](https://discord.gg/SsyY2s2k)

### Open Evaluation Task

The evaluation of CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, and TruthfulQA is conducted using the Opencompass tool.

**Environment setup**

```bash
  conda create --name opencompass python=3.10 
  conda activate opencompass
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  Pip install Faiss-gpu
  cd opencompass && pip install -e .
  cd opencompass/human-eval && pip install -e .
```

**Pre-trained model preparation for track-1**

- [Phi-2](https://huggingface.co/microsoft/phi-2)
- [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Qwen-7B](https://huggingface.co/Qwen/Qwen2-7B)

**Data preparation**

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

**Evaluation Huggingface models**

```bash 
CUDA_VISIBLE_DEVICES=0 python run.py --datasets commonseqa_gen longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen truthfulqa_gen --hf-num-gpus 1 --hf-type base --hf-path meta-llama/Meta-Llama-3-8B --debug --model-kwargs device_map='auto' trust_remote_code=True
## --dataset: specify datasets
```
**Evaluate local models**

  - Your local model must be wrapped in the opencompass format. An example can be found in opencompass/opencompass/models/custom_llama.py Refer to (https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html).
  - Prepare the corresponding configure file. An example can  be found in opencompass/configs/example/example.py 

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --datasets commonseqa_gen longbench bbh_gen gsm8k_gen humaneval_gen FewCLUE_chid_gen truthfulqa_gen --hf-num-gpus 1 --hf-type base --models example --debug --model-kwargs device_map='auto' trust_remote_code=True
# --models: specify the local model
```

> \[!TIP\]
>
> -- The wrapped model file (.py) should be placed in the directory: opencompass/opencompass/models.
>
> -- The prepared configure file should be placed in the directory: /opencompass/configs



### GPU Memory Usage and Throughput Measurement

```bash
# Replace the model/tokenizer loader code with your code. DO NOT CHANGE THE HYPER-PARAMETER SETTING.
python EvaluateThroughputAndMemory.py --model_name MODEL_NAME
```

### Compile Model via MLC-MiniCPM
Refer to https://github.com/OpenBMB/mlc-MiniCPM

**Prepare Environment**

Follow https://llm.mlc.ai/docs/deploy/android.html to prepare requirements.

For the **Compile PyTorch Models from HuggingFace**,  conduct the following instructions to install mlc_chat.

```bash
mkdir -p build && cd build
# generate build configuration
python3 ../cmake/gen_cmake_config.py && cd ..
# build `mlc_chat_cli`
cd build && cmake .. && cmake --build . --parallel $(nproc) && cd ..
# install
cd python && pip install -e . && cd ..
```

**Compile Model** DETAILED USAGE INSTRUCTIONS ARE COMING SOON!

put huggingface downloaded model checkpoint into `dist/models`.

```bash
MODEL_NAME=MiniCPM
MODEL_TYPE=minicpm
mlc_chat convert_weight --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}-hf/  -o dist/$MODEL_NAME/
mlc_chat gen_config --model-type ${MODEL_TYPE} ./dist/models/${MODEL_NAME}-hf/ --conv-template LM --sliding-window-size 768 -o dist/${MODEL_NAME}/
mlc_chat compile --model-type ${MODEL_TYPE} dist/${MODEL_NAME}/mlc-chat-config.json --device android -o ./dist/libs/${MODEL_NAME}-android.tar
cd ./android/library
./prepare_libs.sh
cd -
```

### Submissions Requirements:

Please upload all the required materials to a GitHub repository and submit the repository link to us via the submission form (https://forms.gle/S367FfxUDcjSKz1Q9). The repository should contain:

- A .txt file: It contains a shared link for downloading your model checkpoints in the huggingface format (make sure that the saved model can be downloaded via this shared link).

- A .txt file: It contains a shared link for downloading the compiled model (compiled by MLC-MiniCPM) (make sure that the compiled model can be downloaded via this shared link).

- A folder: Include the runnable source code of your method as well as a readme for usage explanation.

- The (wrapped) model definition file (.py) and its configuration file which are required by opencompass for evaluating your local model. 

- A CSV file: All participating teams are required to evaluate their models locally first and submit the results using a .CSV file. It should contain scores of CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA, Throughput, and GPU memory usage. Please generate .CSV file via Generate_CSV.py

**An example of submission format can be found in Submission_Example folder**

