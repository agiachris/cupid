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


function get_dataset_cmd {
    # Optional: Curate training and holdout sets.
    if [[ ${curate_dataset} == 1 ]]; then

        # Check if curation method is valid.
        if [[ "${curation_method}" != "oracle" && "${curation_method}" != "random" && "${curation_method}" != influence* && "${curation_method}" != *similarity && "${curation_method}" != dem* ]]; then
            echo "Curation method ${curation_method} is not supported for dataset curation."
            exit 1
        fi

        # Adjust policy checkpoint paths.
        train_name="${curated_train_date}_train_${policy}_${task}_${seed}-curation_${curation_method}-filter_${filter_ratio}-select_${select_ratio}"
        train_dir="${train_output_dir}/${curated_train_date}/${train_name}"

        # Adjust result load key. 
        eval_policy_loss_load_key="curation=${curation_method}-filter=${filter_ratio}-select=${select_ratio}-${eval_policy_loss_load_key}"
        eval_action_likelihood_load_key="curation=${curation_method}-filter=${filter_ratio}-select=${select_ratio}-${eval_action_likelihood_load_key}"
    fi
}


function eval_rollout_scores {
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

    # Rollout scoring experiment name.
    exp_name="${result_date}_rollout_scores-seed=${exp_seed}"    
    case "${exp_type}" in
        "diffusion")
            # Diffusion parameters.
            eval_policy_loss_load_key="default_diffusion"
            eval_action_likelihood_load_key="default_diffusion"
            ;;
        *)
            # Default parameters.
            eval_policy_loss_load_key="default"
            eval_action_likelihood_load_key="default"
            ;;
    esac
    get_dataset_cmd

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

    # Methods.
    CMD="${CMD} --eval_policy_loss=${eval_policy_loss}"
    CMD="${CMD} --eval_policy_loss_load_key=${eval_policy_loss_load_key}"
    CMD="${CMD} --eval_action_likelihood=${eval_action_likelihood}"
    CMD="${CMD} --eval_action_likelihood_load_key=${eval_action_likelihood_load_key}"

    run_cmd
}


function eval_rollout_scores_over_seeds {
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                eval_rollout_scores
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    eval_rollout_scores
                done
            fi
        done
    done
}


function eval_rollout_scores_with_curation {
    for curation_method in "${curation_methods[@]}"; do
        for i in "${!curation_filter_ratios[@]}"; do
            filter_ratio="${curation_filter_ratios[i]}"
            select_ratio="${curation_select_ratios[i]}"
            eval_rollout_scores_over_seeds
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
script="eval_rollout_scores"
eval_output_dir="data/outputs/eval_save_episodes"
train_output_dir="data/outputs/train"
project="cupid"
device="cpu"

# General experiment params.
overwrite=0
use_half_precision=0
exp_seed=0

# Policies.
eval_seed="N/A"
train_seed="0 1 2"
train_ckpt="latest"
eval_as_train_seed=1

# Methods (Diff-Conv; DDPM loss).
eval_policy_loss=1
eval_action_likelihood=1
exp_type="diffusion"

# Dates.
train_filter=1  # Official: Used for demo filtering experiments.
train_select=0  # Official: Used for demo selection experiments.
if [[ $train_filter == 1 ]]; then
    result_date="default"
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
    curated_train_date="<enter_policy_curated_retraining_date>" # Optional.
elif [[ $train_select == 1 ]]; then
    result_date="default"
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
    curated_train_date="<enter_policy_curated_retraining_date>" # Optional.
else
    echo "Select a dataset setting."
    exit 1
fi


######################## Standard rollout scoring eval. ########################
curate_dataset=0

# Lowdim-state-based experiments.
state="low_dim"
eval_rollout_scores_over_seeds

# Image-state-based experiments.
state="image"
# eval_rollout_scores_over_seeds


######################## Curation rollout scoring eval. ########################
curate_dataset=1

# Scoring strategy.
curation_methods=(
    ## Official influence (CUPID).
    # "influence_sum_official"
    # "influence_quality_official"

    ## Baselines.
    # "oracle"
    # "random"
    # "state_similarity"
    # "demoscore"
    # "deminf"
)
curation_filter_ratios=(
    # Filtering.
    # 0.10
    # 0.25
    # 0.50
    # 0.75
    # 0.90
    # Selection.
    # 0.00
    # 0.00
    # 0.00
    # 0.00
    # 0.00
)
curation_select_ratios=(
    # Filtering.
    # 0.00
    # 0.00
    # 0.00
    # 0.00
    # 0.00
    # Selection.
    # 0.10
    # 0.25
    # 0.50
    # 0.75
    # 0.90
)

# Lowdim-state-based experiments.
state="low_dim"
# train_policy_with_curation

# Image-state-based experiments.
state="image"
# train_policy_with_curation