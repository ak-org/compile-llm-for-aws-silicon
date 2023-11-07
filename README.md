# How to compile llama-2-7b-chat for inf2.xlarge using Neuron SDK 2.15.0

## Introduction
This repository show how to compile llama-2-7b-chat model for neuron cores using neuron SDK 2.15.0. The compilation process depends on the value of environment variable `NEURON_RT_NUM_CORES`. Currently it is set to 2 to match the number of cores in the inf2.xlarge instance type.

## Pre-requisite

- Make sure that you have appropriate service quota set for the in2.xlarge endpoint 
- You need a valid huggingface account and you must requst access for meta-llama/llama-2-7b-chat model from Meta on the huggingface portal. Once your request is approved, please generate huggingface tokens in your account setting. You will need to setup environment variable and set this token as its value.
- You will need to set your AWS credentials in the following environment variables:

  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

## High level steps

After you have set the huggingface token and AWS credential environment variabels properly:

- Run `scripts/split_and_save.py` to download model from hugginface, split the model weights for neuron core and save it locally
- Run `scripts/compile.py` to compile the model for neuron core

- Run `smep/deploy.py` to deploy the neuron compiled model as sagemaker endpoint on inf2.xlarge instance type
- Run `smep/infer.py` to run inferences against the sagemaker endpoint deployed in the previous step




