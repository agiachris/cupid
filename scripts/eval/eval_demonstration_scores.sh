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


function eval_demonstration_scores {
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

    # Demonstration scoring experiment name.
    exp_name="${result_date}_demonstration_scores-seed=${exp_seed}"

    # Setup.
    CMD="python ${script}.py"
    CMD="${CMD} --exp_name=${exp_name}"
    CMD="${CMD} --eval_dir=${eval_dir}"
    CMD="${CMD} --train_dir=${train_dir}"
    CMD="${CMD} --train_ckpt=${train_ckpt}"
    CMD="${CMD} --result_date=${result_date}"
    CMD="${CMD} --overwrite=${overwrite}"
    CMD="${CMD} --device=${device}"
    CMD="${CMD} --seed=${exp_seed}"
    CMD="${CMD} --use_half_precision=${use_half_precision}"
    CMD="${CMD} --compute_holdout=${compute_holdout}"

    # Methods.
    CMD="${CMD} --eval_offline_policy_loss=${eval_offline_policy_loss}"
    CMD="${CMD} --eval_offline_action_diversity=${eval_offline_action_diversity}"
    CMD="${CMD} --eval_offline_state_diversity=${eval_offline_state_diversity}"
    CMD="${CMD} --eval_online_state_similarity=${eval_online_state_similarity}"
    CMD="${CMD} --eval_online_demo_score=${eval_online_demo_score}"
    CMD="${CMD} --eval_online_trak_influence=${eval_online_trak_influence}"
    
    run_cmd
}


function eval_demonstration_scores_over_seeds {
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                eval_demonstration_scores
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    eval_demonstration_scores
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


######################## Experiment setup. ########################
script="eval_demonstration_scores"
eval_output_dir="data/outputs/eval_save_episodes"
train_output_dir="data/outputs/train"
project="cupid"
device="cpu"

# General experiment params.
overwrite=0
use_half_precision=0
compute_holdout=1
exp_seed=0

# Policies.
eval_seed="N/A"
train_seed="0 1 2"
train_ckpt="latest"
eval_as_train_seed=1

# Methods.
eval_offline_policy_loss=0
eval_offline_action_diversity=0
eval_offline_state_diversity=0
eval_online_state_similarity=0
eval_online_demo_score=0
eval_online_trak_influence=1

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


######################## Standard demo scoring eval. ########################

# Lowdim-state-based experiments.
state="low_dim"
eval_demonstration_scores_over_seeds

# Image-state-based experiments.
state="image"
# eval_demonstration_scores_over_seeds