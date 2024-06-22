import os
os.environ['HF_DATASETS_CACHE'] = '/scratch-shared/HTJ/'
os.environ['HF_TOKENIZERS_CACHE'] = '/scratch-shared/HTJ/tokenizes'
os.environ['HF_HOME'] = '/scratch-shared/HTJ/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/scratch-shared/HTJ/metrics'
os.environ['HF_MODULES_CACHE'] = '/scratch-shared/HTJ/modules'
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Measure GPU memory and throughput of LLM inference")
parser.add_argument('--model_name', type=str, default="gpt2", help="Name of the model to use")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference")
parser.add_argument('--num_repeats', type=int, default=100, help="Number of times to repeat the inference for averaging")
args = parser.parse_args()

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Load a sample of the Wiki dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Function to get GPU memory usage
def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], 
        encoding="utf-8"
    )
    gpu_memory = int(result.strip().split('\n')[0])
    return gpu_memory

# Function to perform inference
def model_inference(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)

# Select a batch of data
sample_batch = encoded_dataset.select(range(args.batch_size))
inputs = {key: torch.tensor(val).cuda() for key, val in sample_batch.to_dict().items() if key in tokenizer.model_input_names}

# Initialize lists to store memory usage and inference time
memory_usages = []
inference_times = []

# Repeat the measurement
for _ in range(args.num_repeats):
    torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
    
    # Measure inference time
    start_time = time.time()
    model_inference(model, inputs)
    end_time = time.time()
    
    # Measure GPU memory usage after inference
    gpu_memory = get_gpu_memory_usage()

    # Calculate inference time
    inference_time = end_time - start_time

    # Store the results
    memory_usages.append(gpu_memory)
    inference_times.append(inference_time)

# Calculate averages
average_memory_usage = np.mean(memory_usages)
average_inference_time = np.mean(inference_times)
throughput = args.batch_size / average_inference_time

print(f"Average memory used during inference: {average_memory_usage} MB")
print(f"Average inference time: {average_inference_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} inferences/second")
