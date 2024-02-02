export CUDA_VISIBLE_DEVICES=0

export WANDB_DISABLED=True
export MDL=meta-llama/Llama-2-7b-hf
export CONFIG=yamls/restart_7b_rosa_gsm_4bit.yaml
export BASE_SAVE_PATH=./checkpoints

export NUM_EPOCHS=1
export LR=2e-4
export WARMUP=20
export BS=32
export PER_DEVICE_BS=1
export SPA_DENSITY=0.006
export SPA_NUM_GRADS=1
export SPA_GRAD_ACC_MODE=mean_squared
export LORA_R=16
export LORA_ALPHA=16
export SCHEDULE=wl64
export SEED=42
export LORA_LR=7e-4

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

export MAX_DURATION=${NUM_EPOCHS}ep

export RUN_NAME=7b_gsm8k-qrosa_${SCHEDULE}_d${SPA_DENSITY}_${SPA_NUM_GRADS}grads_${SPA_GRAD_ACC_MODE}_r${LORA_R}_loralr${LORA_LR}_alpha${LORA_ALPHA}-lr${LR}-epochs${NUM_EPOCHS}-wu${WARMUP}-seed${SEED}-$RANDOM

mkdir -p ${BASE_SAVE_PATH}/masks/
mkdir -p ${BASE_SAVE_PATH}/models/

if [[ "$SPA_DENSITY" != "0" ]]
then

    if [[ $LORA_R == 0 ]]
    then
        export SCHEDULE=spa_only
    fi

    composer train.py \
        ${CONFIG} \
        model_name_or_path=${MDL} \
        max_duration=${MAX_DURATION} \
        run_name=${RUN_NAME} \
        optimizer.lr=${LR} \
        global_train_batch_size=${BS} \
        device_train_microbatch_size=${PER_DEVICE_BS} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        scheduler.t_warmup=${WARMUP}ba \
        rosa.spa_d=${SPA_DENSITY} \
        rosa.spa_num_grads=${SPA_NUM_GRADS} \
        rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
        rosa.lora_r=${LORA_R} \
        rosa.lora_alpha=${LORA_ALPHA} \
        rosa.lora_lr=${LORA_LR} \
        rosa.schedule=${SCHEDULE} \
        global_seed=${SEED} \
        seed=${SEED} \
        hf_save_path=${BASE_SAVE_PATH}/models/ \
        rosa.mask_save_path=${BASE_SAVE_PATH}/masks/${RUN_NAME} \
        rosa.terminate_after_mask_generation=true
fi



export MASK_LOAD_PATH=${BASE_SAVE_PATH}/masks/${RUN_NAME}

if [[ "$SPA_DENSITY" != "0" && $LORA_R -ne 0 ]]
then
    export SCHEDULE=df
elif [[ $LORA_R -ne 0 ]]
then
    export SCHEDULE=lora_only
    export MASK_LOAD_PATH=
else
    export SCHEDULE=spa_only
fi

composer train.py \
    ${CONFIG} \
    model_name_or_path=${MDL} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    scheduler.t_warmup=${WARMUP}ba \
    rosa.spa_d=${SPA_DENSITY} \
    rosa.spa_num_grads=${SPA_NUM_GRADS} \
    rosa.grad_acc_mode=${SPA_GRAD_ACC_MODE} \
    rosa.lora_r=${LORA_R} \
    rosa.lora_alpha=${LORA_ALPHA} \
    rosa.lora_lr=${LORA_LR} \
    rosa.schedule=${SCHEDULE} \
    global_seed=${SEED} \
    seed=${SEED} \
    hf_save_path=${BASE_SAVE_PATH}/models/ \
    rosa.mask_load_path=${MASK_LOAD_PATH}

bash eval_gsm_4bit.sh MDL=${RUN_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}