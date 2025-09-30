import re
import os
import json
import torch
import shutil
import random
import string
import argparse
import switch_generation
from tqdm import tqdm
from peft import LoraConfig
from collections import Counter
from multiprocessing import Pool
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input .jsonl file name")
    parser.add_argument("--gpu_ids", type=str, help="GPU IDs to use, e.g., '0,1,2,3'")
    parser.add_argument("--overide_selector_path", type=str, help="Path to the selector model")
    parser.add_argument("--total_max_length", type=int, default=512, help="Total max length for generation")
    parser.add_argument("--segment_len", type=int, default=50, help="Segment length for generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--objective_flag", type=str, default=False, help="is this an objective task?")

    args = parser.parse_args()
    input_file = args.input
    total_max_length = args.total_max_length
    segment_len = args.segment_len
    batch_size = args.batch_size
    OBJECTIVE_FLAG = args.objective_flag
    gpu_ids = []
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = [0,1,2,3]

    selector_model_path = args.overide_selector_path

    # candidate models
    model_paths = ["meta-llama/Llama-3.1-8B", 
        "allenai/Llama-3.1-Tulu-3-8B-SFT", 
        "allenai/Llama-3.1-Tulu-3-8B"]

    switch_generation.load_models(model_paths, gpu_ids[0:3])
    switch_generation.load_selector_model(selector_model_path, gpu_ids[3])

    data = []
    with open(input_file, "r") as f:
        for line in f:
            example = json.loads(line)
            data.append(example)

    inputs = [x["input"] for x in data]
    outputs, generation_logs = switch_generation.switch_generation(inputs, batch_size=batch_size, total_max_length=total_max_length, objective_flag=OBJECTIVE_FLAG, max_length_per_segment=segment_len)

    assert len(data) == len(inputs) == len(outputs) == len(generation_logs)

    # Save the results
    for i in range(len(data)):
        data[i]["output"] = outputs[i]
        data[i]["generation_log"] = generation_logs[i]
    output_file = input_file.replace(".jsonl", f"_switch_generation.jsonl")
    with open(output_file, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")
    print(f"Results saved to {output_file}")
