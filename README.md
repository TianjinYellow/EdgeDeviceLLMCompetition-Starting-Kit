<h2 align="center">Starting Kit for Edge-Device LLM Competition, NeurIPS 2024</h2>

This is the starting kit for the Edge-Device LLM Competition, a NeurIPS 2024 competition. To learn more about the competition, please see the [competition website](https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/).  This starting kit provides instructions on downloading data, running evaluations, and generating submissions.

<span style="color:red"><strong>Please join us on Discord for discussions and up-to-date announcements:</strong></span>

[https://discord.gg/yD89SPNr3b](https://discord.gg/yD89SPNr3b)


### Evaluation for CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA Tasks

### Open Evaluation Task

The evaluation of CommonsenseQA, BIG-Bench Hard, GSM8K, HumanEval, CHID, and TruthfulQA is conducted using the OpenCompass tool.

**Environment setup**

```bash
  conda create --name opencompass python=3.10 
  conda activate opencompass
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install faiss-gpu

  # Install from source 
  git clone https://github.com/open-compass/opencompass opencompass
  cd opencompass
  git checkout 0.3.1
  pip install -e .

  # or with pip 
  pip install opencompass==0.3.1

  # Install human-eval
  pip install git+https://github.com/open-compass/human-eval.git
```


**Pretrained Model Preparation for Track-1**

- [Phi-2](https://huggingface.co/microsoft/phi-2)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

**Data Preparation(Option-1)**

If your environment cannot access the Internet, you can manually download the dataset.

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```
**Data Preparation(Option-2)**

The OpenCompass will automatically download the datasets either from its own server or from HuggingFace.

**Evaluation Huggingface Models**

- Evaluate with 1-GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
opencompass --datasets commonsenseqa_7shot_cot_gen_734a22 \ 
  FewCLUE_chid_gen \ 
  humaneval_gen \
  bbh_gen \
  gsm8k_gen \ 
  truthfulqa_gen \
  --hf-type chat \
  --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-kwargs device_map='auto' trust_remote_code=True \
  --max-out-len 1024 \
  --debug \ 
  -r latest # You can add --dry-run to auto-download the datasets first before your evaluation
  # for Qwen2-7B-Instruct
  # --hf-path Qwen/Qwen2-7B-Instruct

```

- Evaluate with 8-GPU

```bash
opencompass --datasets commonsenseqa_7shot_cot_gen_734a22 \
  FewCLUE_chid_gen \
  humaneval_gen \
  bbh_gen \
  gsm8k_gen \
  truthfulqa_gen \
  --hf-type chat \
  --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-kwargs device_map='auto' trust_remote_code=True \
  --max-num-workers 8 \
  --max-out-len 1024 \
  -r latest
  # for Qwen2-7B-Instruct
  # --hf-path Qwen/Qwen2-7B-Instruct
```
> \[!Note\]
>
> If there is still an OOM issue when you run it on your server, please set --batch-size to a small value  such as 4 or 2. The default is 8.

**Reference Performance** 


```bash
dataset                                      version    metric            mode      Meta-Llama-3.1-8B-Instruct_hf    Qwen2-7B-Instruct_hf
-------------------------------------------  ---------  ----------------  ------  -------------------------------  ----------------------
commonsense_qa                               734a22     accuracy          gen                               72.89                   33.58
chid-test                                    211ee7     accuracy          gen                               69.43                   81.72
openai_humaneval                             8e312c     humaneval_pass@1  gen                               68.29                   78.05
gsm8k                                        1d7fe4     accuracy          gen                               84.38                   81.12
truthful_qa                                  5ddc62     bleu_acc          gen                                0.28                    0.23
bbh                                          -          naive_average     gen                               67.92                   64.09
```

**Evaluate local models**

  - Your local model must be wrapped in the opencompass format. An example can be found in opencompass/opencompass/models/custom_llama.py Refer to (https://opencompass.readthedocs.io/en/latest/advanced_guides/new_model.html).
  - Prepare the corresponding configuration file. An example can  be found in opencompass/configs/example/example.py NOTE: The path of the saved model weights needs to specified in this configuration file.


> \[!TIP\]
>
> -- The wrapped model file (.py) needs to be placed under the folder: opencompass/opencompass/models.
>
> -- The prepared configuration file needs be placed under the folder: /opencompass/configs.


### GPU Memory Usage and Throughput Measurement

```bash
# Replace the model/tokenizer loader code with your code. DO NOT CHANGE THE HYPER-PARAMETER SETTING.
python EvaluateThroughputAndMemory.py --model_name MODEL_NAME
```
> \[!Note\]
>
> -- batch_size needs to be set to 1 and max_length needs to be set to 2K.

### Compile Model via MLC-MiniCPM 
**A Step by Step instruction are presented in the following document:**
- [Document-English](Step_by_step_instruction_MLC-miniMPC_English.pdf)
- [Document-Zh](Step-by-Step_Instruction_MLC-LLM_zh.pdf)

> \[!Note\]
>
> The download link in Document-Zh has been updated. If the link in Document-English is inaccessible, please use the link provided in Document-Zh.


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

**Compile Model** Refer to https://github.com/OpenBMB/mlc-MiniCPM

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

After synchronizing with our evaluation team, thefollowing files must be included in yoursubmission to allow us successfully evaluatingyour models, for both tracks:

- (1) The configuration and checkpoints of theoriginal HuggingFace model (able to run usingPython and Transformer),

- (2) Code for converting the model to MLC(custom code of model network structure in"convert_weight" and "gen_config"),

- (3)Converted MLC model files (model that canrun normally using the official script),

- (4)APK file (APK file successfully compiledaccording to the official tutorial),

- (5)Script to package the MLC model file into theAPK (script that can successfully package themodel onto an Android device and run it, i.e.,.bundle_weight.py).

- (6) lf you successfully run the APK, please take ascreenshot showing the result and upload it to thedesignated folder. lf it doesn't run successfully,you can document the reasons for the failure in anerror.txt TXT and save it in the folder. We will tryto solve it after the submission deadline.


Please upload all the required materials to a GitHub repository and submit the repository link to us via the [submission form](https://forms.gle/S367FfxUDcjSKz1Q9). The repository should contain:

- A .txt file: It contains a shared link for downloading your model checkpoints in the huggingface format (make sure that the saved model can be downloaded via this shared link).

- A .txt file: It contains a shared link for downloading the compiled model (compiled by MLC-MiniCPM) (make sure that the compiled model can be downloaded via this shared link). **The compiled model should include** the following files necessary for running on the Android platform: .apk, mlc-chat-config.json, ndarray-cache.json, params_shard_x.bin, tokenizer.json, tokenizer.model, and tokenizer_config.json.

- A folder: Include the runnable source code of your method as well as a readme for usage explanation.

- The (wrapped) model definition file (.py) and its configuration file which are required by opencompass for evaluating your local model. 

- A CSV file: All participating teams are required to evaluate their models locally first and submit the results using a .CSV file. It should contain scores of CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, CHID, TruthfulQA, Throughput, and GPU memory usage. Please generate .CSV file via Generate_CSV.py

**An example of submission format can be found in Submission_Example folder**

> \[!Note\]
>
>For the github collaborator, please add this account **"edge-llms-challenge"** to your github repo. Make sure we can access this your submission (github repo) using this github account.
