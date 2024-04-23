## run this script from a directory at the same level as 2.18.1 folder 
docker run -v ./2.18.1:/2.18.1 \
        --device /dev/neuron0:/dev/neuron0 \
        --device /dev/neuron1:/dev/neuron1 \
        --device /dev/neuron2:/dev/neuron2 \
        --device /dev/neuron3:/dev/neuron3 \
        --rm -it \
        --entrypoint /bin/bash 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.18.1-ubuntu20.04

## inside container run following commands
## 
## torchserve --start --ncs --model-store /2.18.1/model_store --models Meta-Llama-3-8B-Instruct