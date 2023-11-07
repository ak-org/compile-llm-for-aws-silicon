#!/bin/sh

## split and save the model 
export HUGGING_FACE_HUB_TOKEN=<<Your hugging face access token goes here>>
python split_and_save.py --model_name 'meta-llama/Llama-2-7b-chat-hf' --save_path "../model_store/llama-2-7b-chat/llama-2-7b-chat-split"
python compile.py 
