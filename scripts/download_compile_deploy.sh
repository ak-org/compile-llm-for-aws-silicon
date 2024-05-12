#!/bin/sh
set -e
# download model from HuggingFace -> compile it for Neuron -> deploy on SageMaker
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
region=$7
role=$8
model_id_wo_repo=`basename $2`
model_id_wo_repo_split=$model_id_wo_repo-split
local_dir=neuron_version/$neuron_version/$model_store/$model_id_wo_repo/$model_id_wo_repo_split
export HF_TOKEN=$token

echo model_id=$model_id, local_dir=$local_dir, neuron_version=$neuron_version, model_store=$model_store, s3_bucket=$s3_bucket, prefix=$prefix, region=$region, role=$role

# download the model
echo going to download model_id=$model_id, local_dir=$local_dir
python scripts/split_and_save.py --model-name $model_id --save-path $local_dir
echo model download step completed

#"../2.18/model_store/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-split/"
# compile the model
echo starting model compilation...
python scripts/compile.py compile --action compile --batch-size 8 --num-neuron-cores 8 --model-dir $local_dir
echo done with model compilation

# now upload the model binaries to the s3 bucket
echo going to upload from neuron_version/$neuron_version/$4/ to s3://$s3_bucket/$prefix/
aws s3 cp --recursive neuron_version/$neuron_version/$model_store/ s3://$s3_bucket/$prefix/
echo done with s3 upload

# dir for storing model artifacts
model_dir=smep-with-lmi/models/$model_id
mkdir -p $model_dir
# prepare serving.properties
serving_prop_fpath=$model_dir/serving-inf2.properties
cat << EOF > $serving_prop_fpath
engine=Python
option.entryPoint=djl_python.transformers_neuronx
option.model_id=s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/${model_id_wo_repo_split}/
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
model_packaging_script_fpath=$model_dir/package-inf2.sh
cat << EOF > $model_packaging_script_fpath
mkdir mymodel
cp serving-inf2.properties  mymodel/serving.properties
tar czvf mymodel-inf2.tar.gz mymodel/
rm -rf mymodel
aws s3 cp mymodel-inf2.tar.gz s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/${model_id_wo_repo_split}/code/
EOF
chmod +x $model_packaging_script_fpath

# now change director to the model dir we just created and run
# the above model packaging script which creates a model.tar.gz that has
# the serving.properties which in turn contains the model path in s3 and 
# other model parameters.
cd $model_dir
echo now in `pwd`
./package-inf2.sh
cd -
echo now back in `pwd`

# all set to deploy the model now
python smep-with-lmi/deploy.py --device inf2 \
  --aws-region $region \
  --role-arn $role \
  --bucket $s3_bucket \
  --model-id $model_id \
  --prefix $prefix \
  --model-s3-uri s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/${model_id_wo_repo_split}/code/mymodel-inf2.tar.gz \
  --neuronx-artifacts-s3-uri s3://${s3_bucket}/${prefix}/${model_id_wo_repo}/neuronx_artifacts

echo all done

