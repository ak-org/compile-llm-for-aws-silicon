from transformers_neuronx.llama.model import LlamaForSampling
from transformers import AutoTokenizer
import torch
import time
import os 
import torch_neuronx
import sys 

# we will pin cores to 2 for inf2.xlarge 
os.environ['NEURON_RT_NUM_CORES'] = '2'
os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf
model_dir = "../2.15.0/model_store/llama-2-7b-chat/llama-2-7b-chat-split"
model_compiled_dir="../2.15.0/model_store/llama-2-7b-chat/neuronx_artifacts"
if sys.argv[1] == "compile":
    start = time.time()
    model = LlamaForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp='f16',
            )
    model.to_neuron()
    # save model to the disk
    model.save("../2.15.0/model_store/llama-2-7b-chat/neuronx_artifacts")
    elapsed = time.time() - start
    print(f'\nCompilation and loading took {elapsed} seconds\n')
elif sys.argv[1] == "infer":
    print('\n Loading pre-compiled model\n')
    ## load model from the disk
    start = time.time()
    model = LlamaForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp='f16',
            )
    model.load(model_compiled_dir)
    model.to_neuron()
    elapsed = time.time() - start
    print(f'\n Model successfully loaded in {elapsed} seconds')
    # construct a tokenizer and encode prompt text
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.add_special_tokens(
            {
            
                "pad_token": "<PAD>",
            }
        )
    prompt = """{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128}}"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(input_ids, sequence_length=2048, top_k=50)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\ngenerated sequences {generated_sequences} in {elapsed} seconds\n')

    prompt = """{"inputs":"Can you explain the concept of photosynthesis in a simple and easy-to-understand way?","parameters":{"max_new_tokens":512}}"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(input_ids, sequence_length=512, top_k=50)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\nGenerated sequences {generated_sequences} in {elapsed} seconds\n')
else:
    print(f'\n**Missing paramter: Specify compiler or infer**\n')
