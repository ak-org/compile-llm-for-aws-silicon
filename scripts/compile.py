from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx import LlamaForSampling, NeuronConfig, GQA, QuantizationConfig
from transformers_neuronx.config import GenerationConfig 
from transformers import AutoTokenizer
import torch
import time
import os 
import torch_neuronx
import sys 

# we will pin cores to 8 for inf2.24xlarge 
os.environ['NEURON_RT_NUM_CORES'] = '8'
os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf
BATCH_SIZE = 4
CONTEXT_LENGTH = 44 # hard coded for sample prompt
model_dir =  "../2.18/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
model_compiled_dir="../2.18/model_store/Meta-Llama-3-8B-Instruct/neuronx_artifacts"
neuron_config = NeuronConfig(
                    on_device_embedding=False,
                    attention_layout='BSH',
                    fuse_qkv=True,
                    group_query_attention=GQA.REPLICATED_HEADS,
                    quant=QuantizationConfig(),
                    on_device_generation=GenerationConfig(do_sample=True)
              )

if sys.argv[1] == "compile":
    start = time.time()
    model = LlamaForSampling.from_pretrained(
            model_dir,
            batch_size=BATCH_SIZE,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp='f16',
            neuron_config=neuron_config,
            n_positions=4096,
            )
    model.to_neuron()
    # save model to the disk
    model.save(model_compiled_dir)
    elapsed = time.time() - start
    print(f'\nCompilation and loading took {elapsed} seconds\n')
elif sys.argv[1] == "infer":
    inputs = torch.zeros((BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
    # construct a tokenizer and encode prompt text
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    system_prompt = "You are a celebrity chef and your respones are always cheerful and positive"
    user_prompt = "How can I make BBQ chicken wings?"
    prompt = f"""
             <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    for i in range(BATCH_SIZE):
        inputs[i] = input_ids[0]
    print('\n Loading pre-compiled model\n')
    ## load model from the disk
    start = time.time()
    model = LlamaForSampling.from_pretrained(
            model_dir,
            batch_size=BATCH_SIZE,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp='f16',
            neuron_config=neuron_config,
            n_positions=4096,
            )
    model.load(model_compiled_dir)
    model.to_neuron()
    elapsed = time.time() - start
    print(f'\n Model successfully loaded in {elapsed} seconds')


    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(inputs, sequence_length=2048, top_k=50, temperature=0.9)
        elapsed = time.time() - start

    #generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\ngenerated sequences in {elapsed} seconds\n')
    
else:
    print(f'\n**Missing paramter: Specify compiler or infer**\n')
