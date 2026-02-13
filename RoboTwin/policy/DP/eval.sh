#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
DEBUG=False
CONFIG_NAME=${7}
timestamp=${8}


export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --config_name ${CONFIG_NAME} \
    --timestamp ${timestamp}


#bash eval.sh blocks_ranking_rgb demo_randomized demo_clean 50 0 1 robot_dp_14 "'20260126-080154'"
#bash eval.sh turn_switch demo_clean demo_clean 50 0 0 train_diffusion_transformer_hybrid_workspace "'20260106-143723'"