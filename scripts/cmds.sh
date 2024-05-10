#!/bin/sh

# download and compile a model from HuggingFace
# param 1: HuggingFace token
# param 2: HuggingFace model id, for example meta-llama/Meta-Llama-3-8B-Instruct
# param 3: local directory path to save the model

## split and save the model 
token=$1
model_id=$2
local_dir=$3
export HF_TOKEN=$token

echo model_id=$model_id, local_dir=$local_dir

# download the model
python split_and_save.py --model_name $model --save_path $local_dir

#"../2.18/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
# compile the model
python compile.py compile 
