import os
import torch
import argparse
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_REPO: str = "meta-llama"
MODEL_ID: str = "Meta-Llama-3-8B-Instruct"
NEURON_VER: str = "2.18.1"

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
        print('Hugging Face Hub token is missing')
        exit(-1)

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", "-m", 
        type=str, 
        default=f"{MODEL_REPO}/{MODEL_ID}",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save-path", "-s",
        type=str,
        default=f"../{NEURON_VER}/model_store/{MODEL_ID}/{MODEL_ID}-split/",
        help="Output directory for downloaded model files",
    )
    args = parser.parse_args()
    print(f"args={args}")

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
