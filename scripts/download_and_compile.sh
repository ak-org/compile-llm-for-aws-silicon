#!/bin/sh
set -e
# download and compile a model from HuggingFace
# param 1: HuggingFace token
# param 2: HuggingFace model id, for example meta-llama/Meta-Llama-3-8B-Instruct
# param 3: local directory path to save the model

## split and save the model 
token=$1
model_id=$2
neuron_version=$3
model_store=$4
s3_bucket=$5
prefix=$6
local_dir=$3/$4/$2
export HF_TOKEN=$token

echo model_id=$model_id, neuron_version=$neuron_version, model_store=$model_store, s3_bucket=$s3_bucket, prefix=$prefix

# download the model
echo going to download model_id=$model_id, local_dir=$local_dir
#python scripts/split_and_save.py --model-name $model_id --save-path $local_dir
echo moel download step completed

#"../2.18/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
# compile the model
echo starting model compilation...
#python scripts/compile.py compile $local_dir
echo done with model compilation

# now upload the model binaries to the s3 bucket
echo going to upload from $neuron_version/$4/ to s3://$s3_bucket/$prefix/
#aws s3 cp --recursive $neuron_version/$4/ s3://$s3_bucket/$prefix/
echo done with s3 upload

# prepare serving.properties
serving_prop_fpath=smep-with-lmi/serving-inf2-$model_id.properties
cat << EOF > $serving_prop_fpath
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=s3://${s3_bucket}/${prefix}/${model_id}/${model_id}-split/
option.load_split_model=True
option.tensor_parallel_degree=8
option.n_positions=4096
option.rolling_batch=auto
option.max_rolling_batch_size=4
option.dtype=fp16
option.model_loading_timeout=1200
option.neuron_optimize_level=3
EOF

# prepare model packaging script
model_packaging_script_fpath=smep-with-lmi/package-inf2-$model_id.sh
cat << EOF > $model_packaging_script_fpath
mkdir mymodel
cp serving-inf2-$model_id.properties  mymodel/serving.properties
tar czvf mymodel-inf2.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-inf2.tar.gz s3://${s3_bucket}/${prefix}/${model_id}/code/
EOF
chmod +x smep-with-lmi/package-inf2-$model_id.sh

