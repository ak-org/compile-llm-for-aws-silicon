import os
import sys
import boto3
import logging
import argparse
import sagemaker
from sagemaker import Model
from sagemaker.utils import name_from_base

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        help="Device name, allowed values \"inf2\" or \"gpu\""
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="us-east-1",
        help="AWS region, default=\"us-east-1\"",
    )

    parser.add_argument(
        "--role-arn",
        type=str,
        help="ARN for the role to be used to deploy the model",
    )

    parser.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket name (without s3://) that contains the model binaries",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        help="Model id",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="lmi",
        help="S3 bucket prefix in which model binaries are kept",
    )

    parser.add_argument(
        "--gpu-instance-type",
        type=str,
        default="ml.g5.12xlarge",
        help="GPU instance type for deploying the model",
    )

    parser.add_argument(
        "--inf2-instance-type",
        type=str,
        default="ml.inf2.24xlarge",
        help="Inf2 instance type for deploying the model",
    )

    parser.add_argument(
        "--gpu-image-uri",
        type=str,
        default="763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-deepspeed0.12.6-cu121",
        help="Image URI for the GPU inference container",
    )

    parser.add_argument(
        "--inf2-image-uri",
        type=str,
        default="763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.27.0-neuronx-sdk2.18.1",
        help="Image URI for the Inf2 inference container",
    )

    args = parser.parse_args()
    logger.info(f"args={args}")


    dev = args.device
    aws_region = args.aws_region
    os.environ['AWS_DEFAULT_REGION'] = aws_region
    role = args.role_arn
    bucket_name = args.bucket
    model_id = args.model_id
    prefix = args.prefix

    s3_uri = f"s3://{bucket_name}/{prefix}/{model_id}/code/mymodel-{dev}.tar.gz"
    if dev == 'gpu':        
        instance_type = args.gpu_instance_type
        image_uri = args.gpu_image_uri
    elif dev == 'inf2':
        instance_type = args.inf2_instance_type
        image_uri = args.inf2_image_uri
    else:
        logger.error('Invalid device type')
        sys.exit(-1)

    model_name = f"{model_id}-{dev}".replace("/", "-")
    endpoint_name = sagemaker.utils.name_from_base(model_name)
    logger.info(f"going to deploy model_id={model_id}, endpoint={endpoint_name},\
                  aws_region={aws_region}\
                  role={role},\
                  bucket_name={bucket_name},\
                  prefix={prefix},\
                  instance_type={instance_type},\
                  image_uri={image_uri}")

    boto3_session=boto3.session.Session(region_name=aws_region)
    smr = boto3.client('sagemaker-runtime')
    sm = boto3.client('sagemaker')
    # sagemaker session for interacting with different AWS APIs
    sess = sagemaker.session.Session(boto3_session, 
                                    sagemaker_client=sm, 
                                    sagemaker_runtime_client=smr)
        
    logger.info(f'Deploying on {dev}')
    logger.info("======================================")
    logger.info(f'Will load artifacts from {s3_uri}')
    logger.info("======================================")

    logger.info("======================================")
    logger.info(f'Using Container image {image_uri}')
    logger.info("======================================")
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
    logger.info(model)

    logger.info(f'\nModel deployment initiated on {dev}\nEndpoint Name: {endpoint_name}\n')
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        #volume_size=512, # not allowed for the selected Instance type ml.g5.12xlarge
        model_data_download_timeout=1200, # increase the timeout to download large model
        container_startup_health_check_timeout=1200, # increase the timeout to load large model,
        wait=True,
    )
    logger.info(f'Model deployment on {dev}\nEndpoint Name: {endpoint_name} finished\n')

    
