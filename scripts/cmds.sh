#!/bin/sh

## split and save the model 
export HF_TOKEN=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
python split_and_save.py --model_name 'meta-llama/Meta-Llama-3-8B-Instruct' --save_path "../2.18.1/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
python compile.py compile 
