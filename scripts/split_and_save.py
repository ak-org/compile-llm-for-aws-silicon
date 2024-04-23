import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.module import save_pretrained_split

def create_directory_if_not_exists(path_str: str) -> str:
    """Creates a directory if it doesn't exist, and returns the directory path."""
    if os.path.isdir(path_str):
        return path_str
    elif input(f"{path_str} does not exist, create directory? [y/n]").lower() == "y":
        os.makedirs(path_str)
        return path_str
    else:
        raise NotADirectoryError(path_str)


if __name__ == "__main__":
    
    if 'HF_TOKEN' not in os.environ:
        print('Hugging face Hub token is missing')
        exit(-1)
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-m", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save_path", "-s",
        type=str,
        default="../2.18.1/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/",
        help="Output directory for downloaded model files",
    )
    args = parser.parse_args()

    save_path = create_directory_if_not_exists(args.save_path)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                    low_cpu_mem_usage=True)


    # Save the model
    save_pretrained_split(hf_model, args.save_path)
    print('Model splitted and saved locally')

    # Load and save tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.save_path)
    print('Tokenizer saved locally')
