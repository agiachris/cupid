#!/bin/bash
set -e


DEBUG=1
SLURM_HOSTNAME="<enter_hostname>"
SLURM_SBATCH_FILE="<enter_sbatch_file>"


function run_cmd {
    echo ""
    echo ${CMD}
    if [[ ${DEBUG} == 0 ]]; then
        if [[ `hostname` == "${SLURM_HOSTNAME}" ]]; then
            sbatch "${SLURM_SBATCH_FILE}" "${CMD}"
        else
            eval ${CMD}
        fi
    fi
}


function set_multi_param {
    if [[ "${param_val}" =~ ^[0-9]+$ ]]; then
        CMD="${CMD} ${param_name}=${param_prefix}${param_val}"
    else
        IFS=' ' read -r -a multi_param_vals <<< "${param_val}"
        for param_val in "${multi_param_vals[@]}"; do
            CMD="${CMD} ${param_name}=${param_prefix}${param_val}"
        done
    fi
}


function eval_demo_score {
    # Training directory (policy checkpoint).
    train_name="${train_date}_train_${policy}_${task}_${seed}"
    train_dir="${train_output_dir}/${train_date}/${train_name}"

    # Evaluation directory (episode/rollouts).
    if [[ $eval_as_train_seed == 1 ]]; then
        eval_name="${train_date}_train_${policy}_${task}_${seed}"
    else
        eval_name="${train_date}_train_${policy}_${task}_${eval_seed}"
    fi
    eval_dir="${eval_output_dir}/${eval_date}/${eval_name}/${train_ckpt}"

    # DemoScore experiment name.
    exp_name="${result_date}_demo_score-seed=${exp_seed}"

    # Setup.
    CMD="python ${script}.py"
    CMD="${CMD} --exp_name=${exp_name}"
    CMD="${CMD} --eval_dir=${eval_dir}"
    CMD="${CMD} --train_dir=${train_dir}"
    CMD="${CMD} --train_ckpt=${train_ckpt}"
    CMD="${CMD} --batch_size=${BATCH_SIZE[${task}_${state}]}"
    CMD="${CMD} --overwrite=${overwrite}"
    CMD="${CMD} --device=${device}"
    CMD="${CMD} --seed=${exp_seed}"
    CMD="${CMD} --use_half_precision=${use_half_precision}"
    CMD="${CMD} --compute_holdout=${compute_holdout}"

    # Set DemoScore classifier parameters.
    param_prefix="${train_output_dir}/${train_date}/${train_date}_train_${demo_score}_${task}_${seed}_"
    param_name="--classifier_train_dirs"
    param_val="${classifier_train_seed}"
    set_multi_param    
    
    param_prefix=""
    param_name="--classifier_train_ckpts"
    param_val="${classifier_train_ckpt}"
    set_multi_param

    CMD="${CMD} --classifier_max_val_episodes=${classifier_max_val_episodes}"
    
    run_cmd
}


function eval_demo_score_over_seeds {
    demo_score="${DEMO_SCORE_METHODS[demo_score_${state}]}"
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                eval_demo_score
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    eval_demo_score
                done
            fi
        done
    done
}


# Tasks.
TASKS=(
    ## Official tasks.
    "lift_mh"
    # "square_mh"
    # "transport_mh"

    ## Supported tasks.
    # "can_mh"
    # "can_ph"
    # "lift_ph"
    # "square_ph"
    # "transport_ph"
    # "tool_hang_ph"
    # "pusht"
    
    ## Unsupported tasks.
    # "block_pushing"
    # "kitchen"
)

# Methods.
declare -A DEMO_SCORE_METHODS=(
    ["demo_score_low_dim"]="demo_score_lowdim"
    ["demo_score_image"]="demo_score_image"
)
METHODS=(
    "diffusion_policy_cnn"
)

# Policies.
declare -A POLICIES=(
    # low_dim.
    ["diffusion_policy_cnn_low_dim"]="diffusion_unet_lowdim"

    # image.
    ["diffusion_policy_cnn_image"]="diffusion_unet_image"
)

# Other parameters.
declare -A BATCH_SIZE=(
    # PushT low_dim.
    ["pusht_low_dim"]=128
    # PushT image.
    ["pusht_image"]=64

    # Robomimic low_dim.
    ["can_mh_low_dim"]=128
    ["lift_mh_low_dim"]=128
    ["square_mh_low_dim"]=128
    ["transport_mh_low_dim"]=128
    ["tool_hang_ph_low_dim"]=128
    # Robomimic image.
    ["can_mh_image"]=64
    ["lift_mh_image"]=64
    ["square_mh_image"]=64
    ["transport_mh_image"]=64
    ["tool_hang_ph_image"]=64
)



######################## Experiment setup. ########################
script="eval_demo_score"
eval_output_dir="data/outputs/eval_save_episodes"
train_output_dir="data/outputs/train"
project="cupid"
device="cuda:0"

# General experiment params.
overwrite=0
use_half_precision=1
compute_holdout=1
exp_seed=0

# Policies.
eval_seed="N/A"
train_seed="0 1 2"
train_ckpt="latest"
eval_as_train_seed=1

# DemoScore settings.
classifier_train_seed="0 1 2"
classifier_train_ckpt="latest"
classifier_max_val_episodes=25

# Dates.
train_filter=1  # Official: Used for demo filtering experiments.
train_select=0  # Official: Used for demo selection experiments.
if [[ $train_filter == 1 ]]; then
    result_date="default"
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
elif [[ $train_select == 1 ]]; then
    result_date="default"
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
else
    echo "Select a dataset setting."
    exit 1
fi


######################## Standard DemoScore eval. ########################

# Lowdim-state-based experiments.
state="low_dim"
eval_demo_score_over_seeds

# Image-state-based experiments.
state="image"
# eval_demo_score_over_seeds