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



######################## Standard training. ########################

# Lowdim-state-based experiments.
state="low_dim"
train_policy

# Image-state-based experiments.
state="image"
# train_policy