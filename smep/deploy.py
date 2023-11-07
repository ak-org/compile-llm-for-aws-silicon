import boto3
import sagemaker
from sagemaker import Model, image_uris, serializers, deserializers
import os
from datetime import datetime
MODEL_NAME="smep-inf2-llama2-7b-chat"
boto3_session=boto3.session.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name="us-east-1")
smr = boto3.client('sagemaker-runtime')
sm = boto3.client('sagemaker')
role = 'arn:aws:iam::102048127330:role/service-role/SageMaker-ak-datascientist'  # execution role for the endpoint
instance_type = "ml.inf2.xlarge"
endpoint_name = sagemaker.utils.name_from_base(MODEL_NAME)

sess = sagemaker.session.Session(boto3_session, sagemaker_client=sm, sagemaker_runtime_client=smr)  # sagemaker session for interacting with different AWS APIs
region = sess._region_name  # region name of the current SageMaker Studio environment
account = sess.account_id()  # account_id of the current SageMaker Studio environment
bucket_name = sess.default_bucket()
prefix='torchserve'
output_path = f"s3://{bucket_name}/{prefix}"
print(f'account={account}, region={region}, role={role}, output_path={output_path}')
s3_uri = f'{output_path}/model_store/llama-2-7b-chat/' #  "s3://sagemaker-us-east-1-102048127330/torchserve/model_store/llama-2-13b-chat/"
print("======================================")
print(f'Will load artifacts from {s3_uri}')
print("======================================")
image_uri = '102048127330.dkr.ecr.us-east-1.amazonaws.com/neuronx:2-14-1'


model = Model(
    name=MODEL_NAME + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    # Enable SageMaker uncompressed model artifacts
    model_data={
        "S3DataSource": {
                "S3Uri": s3_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
        }
    },
    image_uri=image_uri,
    role=role,
    sagemaker_session=sess,
    env={"TS_INSTALL_PY_DEP_PER_MODEL": "true"},
)
print(model)

model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    volume_size=512, # increase the size to store large model
    model_data_download_timeout=3600, # increase the timeout to download large model
    container_startup_health_check_timeout=600, # increase the timeout to load large model
)
