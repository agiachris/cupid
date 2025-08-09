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


function get_num_ckpts_per_seed {
    # Number of checkpoints per training seed.
    if [[ "${train_ckpt}" =~ ^[0-9]+$ ]]; then
        num_ckpts_per_seed=1
    else
        IFS=' ' read -r -a _train_ckpts <<< "${train_ckpt}"
        num_ckpts_per_seed=${#_train_ckpts[@]}
    fi
}


function get_demoscore_train_seed {
    get_num_ckpts_per_seed
    # Bug note: Cannot do arithmetic with train_seed and ckpt directly.
    demoscore_train_seed=0
    for ((i=0; i<${seed}; i++)); do
        let demoscore_train_seed=$demoscore_train_seed+${num_ckpts_per_seed}
    done
    for ((i=0; i<${ckpt}; i++)); do
        let demoscore_train_seed=$demoscore_train_seed+1
    done
}


function get_dataset_cmd {
    DATASET_CMD=""
    
    ## DemoScore dataset (the "dataset" in config).
    
    # Random seed.
    DATASET_CMD="${DATASET_CMD} task.dataset.seed=${seed}"
    DATASET_CMD="${DATASET_CMD} task.dataset.dataset_path=${policy_eval_dir}/episodes"
    
    # Dataset splits.
    DATASET_CMD="${DATASET_CMD} task.dataset.val_ratio=${demo_score_val_ratio}"
    DATASET_CMD="${DATASET_CMD} task.dataset.max_train_episodes=${demo_score_max_train_episodes}"
    
    ## Policy dataset (the "demo_dataset" in config).

    # Random seed.
    DATASET_CMD="${DATASET_CMD} task.demo_dataset.seed=${seed}"

    # Dataset splits.
    case "${task}" in
        lift_mh|can_mh|square_mh|transport_mh|tool_hang_ph)
            # Specify split ratios for RoboMimic tasks.
            DATASET_CMD="${DATASET_CMD} task.demo_dataset.val_ratio=${robomimic_val_ratio}"
            DATASET_CMD="${DATASET_CMD} +task.demo_dataset.dataset_mask_kwargs.train_ratio=${robomimic_train_ratio}"
            # Optional uniform quality subsampling for MH tasks.
            if [[ "${task}" == *_mh ]]; then
                DATASET_CMD="${DATASET_CMD} +task.demo_dataset.dataset_mask_kwargs.uniform_quality=${robomimic_uniform_mh_quality}"
            fi
            ;;
        pusht)
            # Specify split ratios for PushT tasks.
            DATASET_CMD="${DATASET_CMD} task.demo_dataset.max_train_episodes=${pusht_max_train_episodes}"
            ;;
        *)
            echo "Specified task ${task} is not supported."
            exit 1
            ;;
    esac
}


function train_demo_score {
    # Training directory (policy checkpoint).
    policy_train_name="${train_date}_train_${policy}_${task}_${seed}"
    policy_train_dir="${train_output_dir}/${train_date}/${train_name}"

    # Evaluation directory (episode/rollouts).
    if [[ $eval_as_train_seed == 1 ]]; then
        policy_eval_name="${train_date}_train_${policy}_${task}_${seed}"
    else
        policy_eval_name="${train_date}_train_${policy}_${task}_${eval_seed}"
    fi
    policy_eval_dir="${eval_output_dir}/${eval_date}/${policy_eval_name}/${ckpt}"

    # Setup.
    get_demoscore_train_seed
    exp_name="${script}_${demo_score}"
    train_name="${train_date}_${exp_name}_${task}_${seed}_${ckpt}"

    CMD="python ${script}.py --config-dir=configs/${state}/${task}/demo_score --config-name=config.yaml"
    CMD="${CMD} name=${exp_name} hydra.run.dir=${output_dir}/${train_date}/${train_name} training.seed=${demoscore_train_seed}"

    # Training.
    CMD="${CMD} training.num_epochs=${NUM_EPOCHS[${task}_${state}]} checkpoint.topk.k=${checkpoint_topk} training.checkpoint_every=${checkpoint_every}"
    
    # Datasets.
    get_dataset_cmd
    CMD="${CMD} ${DATASET_CMD}"
    
    # Logging.
    CMD="${CMD} logging.name=${train_name} logging.group=${train_date}_${exp_name}_${task} logging.project=${project}"
    CMD="${CMD} multi_run.wandb_name_base=${train_name} multi_run.run_dir=${output_dir}/${train_date}/${train_name}"
    run_cmd
}


function train_demo_score_over_ckpts {
    ckpt="${train_ckpt}"
    if [[ ! "${ckpt}" =~ [[:space:]] ]]; then
       train_demo_score 
    else
        IFS=' ' read -r -a ckpts <<< "${ckpt}"
        for ckpt in "${ckpts[@]}"; do
            train_demo_score
        done
    fi
}


function train_demo_score_over_seeds {
    demo_score="${DEMO_SCORE_METHODS[demo_score_${state}]}"
    for task in "${TASKS[@]}"; do
        for method in "${POLICY_METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                train_demo_score_over_ckpts
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    train_demo_score_over_ckpts
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
POLICY_METHODS=(
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
    # PushT low_dim.
    ["pusht_low_dim"]=1001                          # Tuned.
    # PushT image.
    ["pusht_image"]=301                             # Tuned.
    
    # Robomimic low_dim.
    ["lift_mh_low_dim"]=3001                        # Tuned.
    ["can_mh_low_dim"]=3001                         # Tuned.
    ["square_mh_low_dim"]=3001                      # Tuned.
    ["transport_mh_low_dim"]=3001                   # Tuned.
    ["tool_hang_ph_low_dim"]=1801                   # Tuned.
    # Robomimic image.
    ["lift_mh_image"]=301                           # Tuned.
    ["can_mh_image"]=301                            # Tuned.
    ["square_mh_image"]=301                         # Tuned.
    ["transport_mh_image"]=301                      # Tuned.
    ["tool_hang_ph_image"]=301                      # Tuned.
)



######################## Experiment setup. ########################
script="train"
output_dir="data/outputs/${script}"
train_output_dir="data/outputs/train"
eval_output_dir="data/outputs/eval_save_episodes"
project="cupid"

# General experiment params.
checkpoint_topk=3
checkpoint_every=50
exp_seed=0

# Policies.
eval_seed="N/A"
train_seed="0 1 2"
train_ckpt="0 1 2"
eval_as_train_seed=1

# Dates & robomimic dataset settings.
train_filter=1  # Official: Used for demo filtering experiments.
train_select=0  # Official: Used for demo selection experiments.
if [[ $train_filter == 1 ]]; then
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
    robomimic_train_ratio=0.64
    pusht_max_train_episodes=60
elif [[ $train_select == 1 ]]; then
    eval_date="<enter_policy_eval_date>"
    train_date="<enter_policy_train_date>"
    robomimic_train_ratio=0.16
    pusht_max_train_episodes=15
else
    echo "Select a dataset setting."
    exit 1
fi
robomimic_val_ratio=0.04
robomimic_uniform_mh_quality=true

# DemoScore dataset settings.
demo_score_val_ratio=0.05
demo_score_max_train_episodes=25


######################## Standard training. ########################

# Lowdim-state-based experiments.
state="low_dim"
train_demo_score_over_seeds

# Image-state-based experiments.
state="image"
# train_demo_score_over_seeds