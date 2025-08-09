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
    DATASET_CMD=""
    
    # Random seed.
    DATASET_CMD="${DATASET_CMD} task.dataset.seed=${seed}"
    
    # Dataset splits.
    case "${task}" in
        lift_mh|can_mh|square_mh|transport_mh|tool_hang_ph)
            # Specify split ratios for RoboMimic tasks.
            DATASET_CMD="${DATASET_CMD} task.dataset.val_ratio=${robomimic_val_ratio}"
            DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.train_ratio=${robomimic_train_ratio}"
            # Optional uniform quality subsampling for MH tasks.
            if [[ "${task}" == *_mh ]]; then
                DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.uniform_quality=${robomimic_uniform_mh_quality}"
            fi
            ;;
        pusht)
            # Specify split ratios for PushT tasks.
            DATASET_CMD="${DATASET_CMD} task.dataset.max_train_episodes=${pusht_max_train_episodes}"
            ;;
        *)
            echo "Specified task ${task} is not supported."
            exit 1
            ;;
    esac

    # Optional: Curate training and holdout sets.
    if [[ ${curate_dataset} == 1 ]]; then

        # Check if curation method is valid.
        if [[ "${curation_method}" != "oracle" && "${curation_method}" != "random" && "${curation_method}" != influence* && "${curation_method}" != *similarity && "${curation_method}" != dem* && "${curation_method}" != *diversity && "${curation_method}" != *loss ]]; then
            echo "Curation method ${curation_method} is not supported for dataset curation."
            exit 1
        fi

        # Curation kwargs.
        DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.curate_dataset=true"
        DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.curation_config_dir=configs/curation/${state}/${task}"
        DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.curation_method=${curation_method}"
        DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.filter_ratio=${filter_ratio}"
        DATASET_CMD="${DATASET_CMD} +task.dataset.dataset_mask_kwargs.select_ratio=${select_ratio}"
        train_name="${train_name}-curation_${curation_method}-filter_${filter_ratio}-select_${select_ratio}"

        # Epochs to run on curated dataset.
        if [[ ${use_curation_epochs} == 1 && -v CURATION_EPOCHS["${task}_${state}_${filter_ratio}_${select_ratio}"] ]]; then
            curation_epochs="${CURATION_EPOCHS[${task}_${state}_${filter_ratio}_${select_ratio}]}"
            if [[ ${resume_from_default} == 1 ]]; then
                # Already trained on num_epochs. Subtract from curation epochs.
                num_epochs=$((curation_epochs - num_epochs + 1))
            else
                # First time training. Use curation_epochs directly.
                num_epochs="${curation_epochs}"
            fi
        fi
    fi
}


function train_policy {
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            for seed in "${SEEDS[@]}"; do
                exp_name="${script}_${policy}"
                train_name="${date}_${exp_name}_${task}_${seed}"
                num_epochs="${NUM_EPOCHS[${task}_${state}]}"

                # Manually adjust epochs for filtering / selection base policy.                
                if [[ $train_filter == 1 && -v NUM_EPOCHS["${task}_${state}_filter"] ]]; then
                    num_epochs="${NUM_EPOCHS[${task}_${state}_filter]}"
                elif [[ $train_select == 1 && -v NUM_EPOCHS["${task}_${state}_select"] ]]; then
                    num_epochs="${NUM_EPOCHS[${task}_${state}_select]}"
                fi
                get_dataset_cmd

                # Setup.
                CMD="python ${script}.py --config-dir=configs/${state}/${task}/${method} --config-name=config.yaml"
                CMD="${CMD} name=${exp_name} hydra.run.dir=${output_dir}/${date}/${train_name} training.seed=${seed}"
                
                # Training.
                CMD="${CMD} training.num_epochs=${num_epochs} checkpoint.topk.k=${checkpoint_topk}"
                CMD="${CMD} training.checkpoint_every=${checkpoint_every} training.rollout_every=${checkpoint_every}"
                CMD="${CMD} ${DATASET_CMD}"
                
                # Logging.
                CMD="${CMD} logging.name=${train_name} logging.group=${date}_${exp_name}_${task} logging.project=${project}"
                CMD="${CMD} multi_run.wandb_name_base=${train_name} multi_run.run_dir=${output_dir}/${date}/${train_name}"
                run_cmd
            done
        done
    done
}


function train_policy_with_curation {
    for curation_method in "${curation_methods[@]}"; do
        for i in "${!curation_filter_ratios[@]}"; do
            filter_ratio="${curation_filter_ratios[i]}"
            select_ratio="${curation_select_ratios[i]}"
            train_policy
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


# Epochs.
declare -A NUM_EPOCHS=(
    ## Standard.
    # PushT low_dim.
    ["pusht_low_dim"]=1001                          # Tuned.
    # PushT image.
    ["pusht_image"]=301                             # Tuned.
    
    # Robomimic low_dim.
    ["lift_mh_low_dim"]=1001                        # Tuned.
    ["can_mh_low_dim"]=1001                         # Tuned.
    ["square_mh_low_dim"]=1751                      # Tuned.
    ["transport_mh_low_dim"]=1001                   # Tuned.
    ["tool_hang_ph_low_dim"]=601                    # Tuned.
    # Robomimic image.
    ["lift_mh_image"]=301                           # Tuned.
    ["can_mh_image"]=301                            # Tuned.
    ["square_mh_image"]=301                         # Tuned.
    ["transport_mh_image"]=301                      # Tuned.
    ["tool_hang_ph_image"]=301                      # Tuned.

    ## Curation: Selection.
    # Robomimic low_dim.
    ["lift_mh_low_dim_select"]=2501                 # Tuned.
    ["square_mh_low_dim_select"]=3001               # Tuned.
    ["transport_mh_low_dim_select"]=2501            # Tuned.
    # Robomimic image.
    ["lift_mh_image_select"]=401                    # Tuned.
    ["square_mh_image_select"]=401                  # Tuned.
    ["transport_mh_image_select"]=401               # Tuned.
)



######################## Experiment setup. ########################
date="<enter_date>"
script="train"
output_dir="data/outputs/${script}"
project="cupid"

# General experiment params.
SEEDS=(0 1 2)
checkpoint_topk=3
checkpoint_every=50

# Dataset settings.
train_full=0    # Unofficial: Train policy on entire dataset.
train_filter=1  # Official: Used for demo filtering experiments.
train_select=0  # Official: Used for demo selection experiments.
if [[ $train_full == 1 ]]; then
    robomimic_train_ratio=0.96
    pusht_max_train_episodes=90
elif [[ $train_filter == 1 ]]; then
    robomimic_train_ratio=0.64
    pusht_max_train_episodes=60
elif [[ $train_select == 1 ]]; then
    robomimic_train_ratio=0.16
    pusht_max_train_episodes=15
else
    echo "Select a dataset setting."
    exit 1
fi
robomimic_val_ratio=0.04
robomimic_uniform_mh_quality=true


######################## Curated retraining. ########################
curate_dataset=1

# General experiment params.
SEEDS=(0 1 2)
checkpoint_topk=1  # Reduce memory overhead.
checkpoint_every=50

# Additional training.
use_curation_epochs=1
resume_from_default=0
declare -A CURATION_EPOCHS=(
    # Robomimic low_dim: Filtering.
    ["lift_mh_low_dim_0.50_0.00"]=2301
    ["lift_mh_low_dim_0.75_0.00"]=2551
    ["lift_mh_low_dim_0.90_0.00"]=4051

    ["square_mh_low_dim_0.10_0.00"]=1751
    ["square_mh_low_dim_0.25_0.00"]=2001
    ["square_mh_low_dim_0.50_0.00"]=2551
    ["square_mh_low_dim_0.75_0.00"]=3051
    ["square_mh_low_dim_0.90_0.00"]=4051

    ["transport_mh_low_dim_0.50_0.00"]=2301
    ["transport_mh_low_dim_0.75_0.00"]=2551
    ["transport_mh_low_dim_0.90_0.00"]=3051

    # Robomimic low_dim: Selection.
    ["square_mh_low_dim_0.00_0.10"]=2751
    ["square_mh_low_dim_0.00_0.25"]=2501
    ["square_mh_low_dim_0.00_0.50"]=2001
    ["square_mh_low_dim_0.00_0.75"]=1751
    ["square_mh_low_dim_0.00_0.90"]=1501

    ["transport_mh_low_dim_0.00_0.10"]=2251
    ["transport_mh_low_dim_0.00_0.25"]=2001
    ["transport_mh_low_dim_0.00_0.50"]=1501
    ["transport_mh_low_dim_0.00_0.75"]=1251
    ["transport_mh_low_dim_0.00_0.90"]=1001
)


# Scoring strategy.
curation_methods=(
    ## Official influence (CUPID).
    "influence_sum_official"
    "influence_quality_official"

    ## Baselines.
    "oracle"
    "random"
    # "state_similarity"
    # "demoscore"
    # "deminf"

    ## Baselines excluded for poor performance.
    # "state_diversity"
    # "action_diversity"
    # "policy_loss"
)
curation_filter_ratios=(  # Ratio of data to filter from training data.
    # Filtering.
    0.10
    0.25
    0.50
    0.75
    0.90
    # Selection.
    # 0.00
    # 0.00
    # 0.00
    # 0.00
    # 0.00
)
curation_select_ratios=(  # Ratio of data to select from holdout data.
    # Filtering.
    0.00
    0.00
    0.00
    0.00
    0.00
    # Selection.
    # 0.10
    # 0.25
    # 0.50
    # 0.75
    # 0.90
)

# Lowdim-state-based experiments.
state="low_dim"
train_policy_with_curation

# Image-state-based experiments.
state="image"
# train_policy_with_curation