#!/bin/sh

## split and save the model 
## Update HF_TOKEN value in line 5
export HF_TOKEN=hf_OImhboCwtKsmGVLpP
python split_and_save.py --model_name 'meta-llama/Meta-Llama-3-8B-Instruct' --save_path "../2.18.1/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
python compile.py compile 
