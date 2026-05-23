#!/bin/bash
# Usage: nvrc_train.sh <GPU_ID> <VID> <LAMB> <SCALE> <LR_S1> <LR_S2> <GRAD_ACCUM> <BATCH_SIZE> [LOSS_TYPE]
# Defaults: GPU_ID=0, VID=BasketballDrive_1920x1080_50, LAMB=1.0, SCALE=s, LR_S1=2e-3, LR_S2=1e-4, GRAD_ACCUM=8, BATCH_SIZE=80, LOSS_TYPE=wd

GPU_ID=0
VID=BasketballDrive_1920x1080_50
LAMB=1.0
SCALE=s
LR_S1=2e-3
LR_S2=1e-4
T_PATCH=1
H_PATCH=216
W_PATCH=240
GRAD_ACCUM=8
BATCH_SIZE=80
LOSS_TYPE=wd

while [[ $# -gt 0 ]]; do
    case "$1" in
        -gpu | --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        -vid | --vid)
            VID="$2"
            shift 2
            ;;
        -lamb | --lamb)
            LAMB="$2"
            shift 2
            ;;
        -scale | --scale)
            SCALE="$2"
            shift 2
            ;;
        -lr1 | --lr_s1)
            LR_S1="$2"
            shift 2
            ;;
        -lr2 | --lr_s2)
            LR_S2="$2"
            shift 2
            ;;
        -tpatch | --t_patch)
            T_PATCH="$2"
            shift 2
            ;;
        -hpitch | --h_patch)
            H_PATCH="$2"
            shift 2
            ;;
        -wpitch | --w_patch)
            W_PATCH="$2"
            shift 2
            ;;
        -ga | --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        -bs | --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -loss | --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "GPU_ID: ${GPU_ID}"
echo "VID: ${VID}"
echo "LAMB: ${LAMB}"
echo "SCALE: ${SCALE}"
echo "LR_S1: ${LR_S1}"
echo "LR_S2: ${LR_S2}"
echo "GRAD_ACCUM: ${GRAD_ACCUM}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LOSS_TYPE: ${LOSS_TYPE}"

# Map LOSS_TYPE to config file
case "${LOSS_TYPE}" in
    wd|rankdvqa|l1_ms-ssim|l1_ms-ssim-5x5)
        TRAIN_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        EVAL_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        ;;
    *)
        echo "Error: Unknown LOSS_TYPE '${LOSS_TYPE}'"
        echo "  Available: wd, rankdvqa, l1_ms-ssim, l1_ms-ssim-5x5"
        exit 1
        ;;
esac

WORK_DIR=${WORK_DIR:-${HOME}/PerceptualLoss}
ROOT=${WORK_DIR}
COMPRESS_MODEL_CFG_S1=scripts/configs/nvrc/compress_models/nvrc_s1.yaml
COMPRESS_MODEL_CFG_S2=scripts/configs/nvrc/compress_models/nvrc_s2.yaml
MODEL_CFG_S1=scripts/configs/nvrc/models/uvg_hinerv-v2-${SCALE}_1920x1080.yaml
MODEL_CFG_S2=${MODEL_CFG_S1}
EXP_CFG_S1=scripts/configs/nvrc/overfit/s1-360e.yaml
EXP_CFG_S2=scripts/configs/nvrc/overfit/s2-30e.yaml
DATASET_DIR=${WORK_DIR}/Datasets/
DATASET=${VID}
START_FRAME=-1
NUM_FRAMES=-1
INTRA_PERIOD=-1
FMT=png
T=-1
H=-1
W=-1
NUM_PROC=1
GRAD_ACCUM=${GRAD_ACCUM}
TRAIN_BATCH_SIZE=${BATCH_SIZE}
EVAL_BATCH_SIZE=1
MODEL_NAME=nvrc_uvg_hinerv-v2-${SCALE}_1920x1080_x${GRAD_ACCUM}x${TRAIN_BATCH_SIZE}

OUTPUT=${WORK_DIR}/Outputs/NVRC/${DATASET}
EXP_NAME_S1=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S1}_${LOSS_TYPE}_s1
EXP_NAME_S2=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S2}_${LOSS_TYPE}_s2

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Start running the script with experiment:"
echo "    Dataset: ${DATASET}"
echo "    Output: ${OUTPUT}"
echo "    Stage 1: ${EXP_NAME_S1}"
echo "    Stage 2: ${EXP_NAME_S2}"

run_training() {
    local exp_cfg=$1
    local exp_name=$2
    local lr=$3
    local min_lr=$4
    local eval_log=$5

    if [ ! -d ${OUTPUT}/${exp_name} ]; then
        CONDA_ROOT=$(dirname $(dirname $(which conda)))
        . ${CONDA_ROOT}/bin/activate
        conda activate perceptual
        cd $ROOT/NVRC && \
        accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                          --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
        main_nvrc.py --exp-config ${exp_cfg} \
                     --output ${OUTPUT} --exp-name ${exp_name} \
                     --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                     --compress-model-config ${COMPRESS_MODEL_CFG_S1} --model-config ${MODEL_CFG_S1} \
                     --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                     --lamb ${LAMB} \
                     --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                     --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                     --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                     --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                     --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                     --train-enable-log false --eval-enable-log ${eval_log} --log-epochs -2 \
                     --opt adam --sched cosine \
                     --lr ${lr} --warmup-lr 1e-5 --min-lr ${min_lr} --auto-lr-scaling false --max-norm 1.0 \
                     --workers 4 --prefetch-factor 4 \
                     ${6:-}
    else
        echo "Experiment ${exp_name} already exists, skipping."
    fi
}

# S1 training
run_training ${EXP_CFG_S1} ${EXP_NAME_S1} ${LR_S1} 1e-4 false

if [[ ! -f "${OUTPUT}/${EXP_NAME_S1}/results/all.txt" ]]; then
    echo "S1 training not completed, please check the logs."
    exit 1
fi

# S2 training
run_training ${EXP_CFG_S2} ${EXP_NAME_S2} ${LR_S2} 1e-5 true "--resume ${OUTPUT}/${EXP_NAME_S1} --resume-model-only true"

if [ ! -f "${OUTPUT}/${EXP_NAME_S2}/results/all.txt" ]; then
    echo "S2 training not completed, please check the logs."
    exit 1
fi
