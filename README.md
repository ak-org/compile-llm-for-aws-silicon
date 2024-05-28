# How to compile a model available on Hugging Face for AWS Inferentia using the Neuron SDK

## Introduction
This repository shows how to compile Foundation Models (FMs) such as `Meta-Llama-3-8B-Instruct` available on the Hugging Face model hub for Neuron cores using neuron SDK 2.18.1. The compilation process depends on the value of environment variable `NEURON_RT_NUM_CORES`.

## Pre-requisite

1. The Neuron SDK requires that you compile the model on an Inferentia instance. So this code needs to be run on an `Inf2` EC2 instance. The `Meta-Llama-3-8B-Instruct` was compiled on an `inf2.24xlarge` instance.

1. You need a valid Hugging Face token to download gated models from the Hugging Face model hub.

>It is best to use `VSCode` to connect to your EC2 instance as we would be running the code from a `bash` shell.

## High level steps

1. Create an `Inf2` based EC2 instance.
1. Download and install [Conda](https://www.anaconda.com/download#linux) on your EC2 VM.
1. Create a new conda environment for `Python 3.10` and install the packages listed in `requirements.txt`.

    ```{.bash}
    conda create --name awschips_py310 -y python=3.10 ipykernel
    source activate awschips_py310;
    pip install -r requirements.txt
    ```
1. Clone this repo on the EC2 VM.
1. Change directory to the code repo directory.
1. Run the `download_compile_deploy.sh` script using the following command. This script will do a bunch of things:
    1. Download the model from Hugging Face.
    1. Compile the model for Neuron.
    1. Upload the model files to S3.
    1. Create a `settings.properties` file that refers to the model in S3 and create a `model.tar.gz` with the `settings.properties`.
    1. Deploy the model on a SageMaker endpoint.
    ```{.bash}
    # replace the model id, bucket name and role parameters as appropriate
    hf_token=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
    model_id=meta-llama/Meta-Llama-3-8B-Instruct
    neuron_version=2.18
    model_store=model_store
    s3_bucket="llm-models"
    s3_prefix=lmi
    region=us-east-1    
    batch_size=4
    num_neuron_cores=8
    ml_instance_type=ml.trn1.32xlarge
    role="arn:aws:iam::015469603702:role/SageMakerRepoRole"
    ./scripts/download_compile_deploy.sh $hf_token \
     $model_id \
     $neuron_version \
     $model_store \
     $s3_bucket \
     $s3_prefix \
     $region \
     $role \
     $batch_size \
     $num_neuron_cores \
     $ml_instance_type> script.log 2>&1 
    ```
1. The model is deployed now, note the endpoint name from the SageMaker console and you can use it for testing inference via the SageMaker `invoke_endpoint` call as shown in `infer.py` included in this repo, and also, benchmarking performance via the Bring Your Own Endpoint option in [`FMBench`](https://github.com/aws-samples/foundation-model-benchmarking-tool).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.
