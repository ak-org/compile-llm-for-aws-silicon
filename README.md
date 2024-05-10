# How to compile Meta-Llama-3-8B-Instruct for inf2 using Neuron SDK 2.18.1

## Introduction
This repository show how to compile Meta-Llama-3-8B-Instruct model for neuron cores using neuron SDK 2.18.1. The compilation process depends on the value of environment variable `NEURON_RT_NUM_CORES`. Currently it is set to 8 to match the number of cores in the inf2.24xlarge instance type.

## Pre-requisite

- Make sure that you have appropriate service quota set for the in2.24xlarge endpoint 
- You need a valid huggingface account and you must requst access for meta-llama/Meta-Llama-3-8B-Instruct model from Meta on the huggingface portal. Once your request is approved, please generate huggingface tokens in your account setting. You will need to setup environment variable and set this token as its value.


## High level steps

Create a new conda environment for Python 3.10 and install the packages listed in `requirements.txt`.

```{.bash}
conda create --name awschips_py310 -y python=3.10 ipykernel
source activate awschips_py310;
pip install -r requirements.txt
```

After you have set the huggingface token and AWS credential environment variabels properly:

- Run `scripts/split_and_save.py` to download model from hugginface, split the model weights for neuron core and save it locally
- Run `scripts/compile.py` to compile the model for neuron core. The artifacts will be stored in 2.18.1 folder 
- Run `docker\test_locally` to test the container solution locally 
- Run `smep/deploy.py` to deploy the neuron compiled model as sagemaker endpoint on inf2.xlarge instance type
- Run `smep/infer.py` to run inferences against the sagemaker endpoint deployed in the previous step
- Contents of `smep/Meta-Llama-3-8b-instruct` folder contains model metadata and config files required to serve inference using torchserve 




