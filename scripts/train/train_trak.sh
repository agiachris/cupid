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


function get_num_seeds {
    # Number of training seeds.
    if [[ "${train_seed}" =~ ^[0-9]+$ ]]; then
        num_seeds=1
    else
        IFS=' ' read -r -a _train_seeds <<< "${train_seed}"
        num_seeds=${#_train_seeds[@]}
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


function get_num_ckpts {
    # Total number of training checkpoints.
    if [[ -n "${use_num_ckpts}" ]]; then
        num_ckpts="${use_num_ckpts}"
    else
        get_num_seeds
        get_num_ckpts_per_seed
        let num_ckpts=$num_seeds*$num_ckpts_per_seed
    fi
}


function get_model_id {
    if [[ -n "${use_model_id}" ]]; then
        model_id="${use_model_id}"
    else
        get_num_seeds
        get_num_ckpts_per_seed
        # Bug note: Cannot do arithmetic with train_seed and train_ckpt directly.
        model_id=0
        for ((i=0; i<${seed}; i++)); do
            let model_id=$model_id+${num_ckpts_per_seed}
        done
        for ((i=0; i<${ckpt}; i++)); do
            let model_id=$model_id+1
        done
    fi
}


function get_base_cmd {
    # Compute num_ckpts and model_id.
    get_num_ckpts
    get_model_id

    # Training directory (policy checkpoint).
    train_name="${train_date}_train_${policy}_${task}_${seed}"
    train_dir="${train_output_dir}/${train_date}/${train_name}"

    # Evaluation directory (episode/rollouts).
    if [[ $eval_as_train_seed == 1 ]]; then
        eval_name="${train_date}_train_${policy}_${task}_${seed}"
    else
        eval_name="${train_date}_train_${policy}_${task}_${eval_seed}"
    fi
    eval_dir="${eval_output_dir}/${eval_date}/${eval_name}/${eval_ckpt}"

    # Setup.
    CMD=""
    # Experiment params.
    CMD="${CMD} --eval_dir=${eval_dir}"
    CMD="${CMD} --train_dir=${train_dir}"
    CMD="${CMD} --train_ckpt=${train_ckpt}"
    CMD="${CMD} --model_keys=${MODEL_KEYS[${state}]}"
    # TRAK params.
    CMD="${CMD} --modelout_fn=${MODELOUT_FN[${method}_${state}]}"
    CMD="${CMD} --gradient_co=${GRADIENT_CO[${method}_${state}]}"
    CMD="${CMD} --proj_dim=${proj_dim}"
    CMD="${CMD} --proj_max_batch_size=${proj_max_batch_size}"
    CMD="${CMD} --lambda_reg=${lambda_reg}"
    CMD="${CMD} --use_half_precision=${use_half_precision}"
    # Other params.
    CMD="${CMD} --batch_size=${BATCH_SIZE[${task}_${state}]}"
    CMD="${CMD} --device=${device}"
    CMD="${CMD} --seed=${exp_seed}"

    # Experiment name.
    exp_name="${result_date}_trak_results-proj_dim=${proj_dim}-lambda_reg=${lambda_reg}-num_ckpts=${num_ckpts}-seed=${exp_seed}"
}


function get_diffusion_cmd {
    # Diffusion TRAK parameters.
    CMD="${CMD} --loss_fn=${loss_fn}"
    if [[ ${finalize_trak} == 0 ]]; then
        CMD="${CMD} --num_timesteps=${num_timesteps}"
    fi

    # Append to experiment name.
    exp_name="${exp_name}-loss_fn=${loss_fn}-num_timesteps=${num_timesteps}"
}


function train_trak {
    # Get base command.
    get_base_cmd

    case "${exp_type}" in
        "diffusion")
            # Diffusion parameters.
            get_diffusion_cmd
            train_script="${train_script_prefix}_diffusion"
            ;;
        *)
            echo "Not implemented."
            exit 1
            ;;
    esac
    CMD="${CMD} --exp_name=${exp_name}"

    # Featurizing holdout set as well.
    if [[ $featurize_holdout == 1 ]]; then
        CMD="${CMD} --featurize_holdout=true"
    fi

    # Training TRAK.
    if [[ ${finalize_trak} == 0 ]]; then
        CMD="python ${train_script}.py --model_id=${model_id} ${CMD}"

        # Finalize scores after featurizing; no checkpoint ensembling.
        if [[ $num_ckpts == 1 || $finalize_on_train == 1 ]]; then
            CMD="${CMD} --finalize_scores=true"
        fi

        run_cmd

    # Finalizing TRAK.
    else
        CMD="python ${finalize_script}.py --num_ckpts=${num_ckpts} ${CMD}"
        run_cmd
        exit 1
    fi
}


function train_trak_over_ckpts {
    ckpt="${train_ckpt}"
    if [[ ! "${ckpt}" =~ [[:space:]] ]]; then
        train_trak
    else
        IFS=' ' read -r -a ckpts <<< "${ckpt}"
        for ckpt in "${ckpts[@]}"; do
            train_trak
        done
    fi
}


function train_trak_over_seeds {
    for task in "${TASKS[@]}"; do
        for method in "${METHODS[@]}"; do
            policy="${POLICIES[${method}_${state}]}"
            model_ids=""
            
            # Compute over seeds.
            seed="${train_seed}"
            if [[ "${seed}" =~ ^[0-9]+$ ]]; then
                train_trak_over_ckpts    
            else
                IFS=' ' read -r -a train_seeds <<< "${seed}"
                for seed in "${train_seeds[@]}"; do
                    train_trak_over_ckpts
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
declare -A MODEL_KEYS=(
    # low_dim.
    ["low_dim"]="model."
    # image.
    ["image"]="obs_encoder.,model."
)

# Trak parameters.
declare -A MODELOUT_FN=(
    # low_dim.
    ["diffusion_policy_cnn_low_dim"]="DiffusionLowdimFunctionalModelOutput"
    
    # image.
    ["diffusion_policy_cnn_image"]="DiffusionHybridImageFunctionalModelOutput"
)
declare -A GRADIENT_CO=(
    # low_dim.
    ["diffusion_policy_cnn_low_dim"]="DiffusionLowdimFunctionalGradientComputer"
    
    # image.
    ["diffusion_policy_cnn_image"]="DiffusionHybridImageFunctionalGradientComputer"
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
    ["transport_mh_low_dim"]=64
    ["tool_hang_ph_low_dim"]=128
    # Robomimic image.
    ["can_mh_image"]=32
    ["lift_mh_image"]=32
    ["square_mh_image"]=32
    ["transport_mh_image"]=16
    ["tool_hang_ph_image"]=32
)



######################## Experiment setup. ########################
train_script_prefix="train_trak"
finalize_script="finalize_trak"
train_output_dir="data/outputs/train"
eval_output_dir="data/outputs/eval_save_episodes"
project="cupid"
device="cuda:0"

# General experiment params.
use_half_precision=0
featurize_holdout=1
finalize_on_train=1
finalize_trak=0
exp_seed=0

# Policies.
eval_seed="N/A"
eval_ckpt="latest"
train_seed="0 1 2"
train_ckpt="latest" # Supports multi-checkpoint.
eval_as_train_seed=1
use_model_id=0
use_num_ckpts=1

# TRAK settings.
proj_dim=4000
proj_max_batch_size=32
lambda_reg=0.0

# Method: Diff-Conv; Square loss.
exp_type="diffusion"
loss_fn="square" # loss_fn="ddpm" 
num_timesteps=64

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


######################## Standard policy attribution. ########################

# Lowdim-state-based experiments.
state="low_dim"
train_trak_over_seeds

# Image-state-based experiments.
state="image"
# train_trak_over_seeds
