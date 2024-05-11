import boto3
import sagemaker
import os
from sagemaker import Model
from sagemaker.utils import name_from_base
import sys
import boto3
if len(sys.argv) != 2:
    print('Please specify device: gpu or inf2')
    exit(-1)
dev = sys.argv[1]

## example S3 for huggingface models 
## s3://sagemaker-example-files-prod-us-east-1/models/llama-2/fp16/


aws_region = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = aws_region
role = 'arn:aws:iam::015469603702:role/SageMakerRepoRole'  # execution role for the endpoint
bucket_name = 'llm-models'
model_id="Meta-Llama-3-8B-Instruct"
prefix="lmi"

if dev == 'gpu':
    s3_uri = f"s3://{bucket_name}/{prefix}/{model_id}/code/mymodel-{dev}.tar.gz"
    instance_type = "ml.g5.12xlarge"
    image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-deepspeed0.12.6-cu121'
elif dev == 'inf2':
    neuronx_sdk_release = '2.18.1'
    s3_uri = f"s3://{bucket_name}/{prefix}/{model_id}/code/mymodel-{dev}.tar.gz"
    instance_type = "ml.inf2.24xlarge"
    image_uri = f'763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-neuronx-sdk2.18.1'
else:
    print('Invalid device type')
    exit(-1)
MODEL_NAME=f"smep-{dev}-{model_id}"
endpoint_name = sagemaker.utils.name_from_base(MODEL_NAME)

boto3_session=boto3.session.Session(region_name=aws_region)
smr = boto3.client('sagemaker-runtime')
sm = boto3.client('sagemaker')
sess = sagemaker.session.Session(boto3_session, 
                                 sagemaker_client=sm, 
                                 sagemaker_runtime_client=smr)  # sagemaker session for interacting with different AWS APIs
    
print(f'Deploying on {dev}')
print("======================================")
print(f'Will load artifacts from {s3_uri}')
print("======================================")

print("======================================")
print(f'Using Container image {image_uri}')
print("======================================")
model_name = name_from_base(MODEL_NAME)
model = Model(
    name=endpoint_name,
    # Enable SageMaker uncompressed model artifacts
    model_data={
        "S3DataSource": {
                "S3Uri": s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "Gzip",
        }
    },
    image_uri=image_uri,
    role=role,
    env = {
        "NEURON_COMPILE_CACHE_URL": f"s3://{bucket_name}/{prefix}/{model_id}/neuronx_artifacts/"
    },
    sagemaker_session=sess
    #env - set TS_INSTALL_PY_DEP_PER_MODEL to true, if you are using Pytorch serving
    #this will tell server to run requirements.txt to deploy any additional packages
)
print(model)

model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    #volume_size=512, # not allowed for the selected Instance type ml.g5.12xlarge
    model_data_download_timeout=1200, # increase the timeout to download large model
    container_startup_health_check_timeout=1200, # increase the timeout to load large model,
    wait=False,
)

print(f'\nModel deployment initiated on {dev}\nEndpoint Name: {endpoint_name}\n')
