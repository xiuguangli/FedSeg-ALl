# 运行入口：默认训练 VOC；所有变量都可以用环境变量覆盖。
# 例子：
#   EVAL_ONLY=True bash run_voc.sh
#   DATASET=cityscapes ROOT_DIR=data/cityscapes NUM_CLS=19 bash run_voc.sh
#   EVAL_FAST_MODE=False EVAL_BUCKETS="" bash run_voc.sh
date_now=$(date +"%Y%m%d_%H%M%S")
#python=../envs/torch11/bin/python

# ===== 数据集路径 =====
# ROOT_DIR 是当前数据集根目录。默认用 VOC；切 Cityscapes/CamVid/ADE20K 时建议同时改
# DATASET、ROOT_DIR、NUM_CLS，避免类别数和标签映射不一致。
#ROOT_DIR='../data/cityscapes'
#ROOT_DIR='../data/cityscapes_split_erase19'
#ROOT_DIR='../data/cityscapes_split_erase19C2'
ROOT_DIR="${ROOT_DIR:-data/voc}"
#ROOT_DIR='../data/ade20k_erase_150C1'

# ===== 损失与标签生成 =====
# LABEL_ONLINE_GEN=True 会在线生成伪标签，通常和 PSEUDO_LABLE 配合使用。
LABEL_ONLINE_GEN="${LABEL_ONLINE_GEN:-False}"
# LOSSTYPE 支持 ce/ohem/back/dice/focal/lovasz/bce；当前默认 back，沿用原实验设置。
LOSSTYPE="${LOSSTYPE:-back}" # ce,ohem,back,'dice','focal','lovasz','bce'

# ===== 联邦训练规模 =====
# WARMSTEP: warmup 相关步数，具体在训练 loss/调度逻辑中使用。
WARMSTEP="${WARMSTEP:-20}"
# FRAC_NUM: 每轮选择多少个客户端参与训练。
FRAC_NUM="${FRAC_NUM:-5}"
# LOCAL_EP: 每个被选客户端本地训练 epoch 数。
LOCAL_EP="${LOCAL_EP:-2}"
# LOCAL_BS: 客户端本地 batch size，显存不够时优先调小它。
LOCAL_BS="${LOCAL_BS:-8}"
# MIXLABLE: 是否启用 mix label 相关逻辑。
MIXLABLE="${MIXLABLE:-True}"
# FEDPROX_MU: FedProx 正则强度；0 表示基本等价于不用 FedProx 项。
FEDPROX_MU="${FEDPROX_MU:-0}"

# ===== 蒸馏相关 =====
# DISTILL=False 时下面几个蒸馏温度/权重参数不会起主要作用。
DISTILL="${DISTILL:-False}"
# TEMP_DIST: 蒸馏 softmax 温度。
TEMP_DIST="${TEMP_DIST:-0.1}"
# LAMB_PI / LAMB_PA: 蒸馏损失中 pi/pa 两部分的权重。
LAMB_PI="${LAMB_PI:-0.1}"
LAMB_PA="${LAMB_PA:-0}"

# ===== 初始化 =====
# RAND_INIT=True 表示随机初始化；False 时会走当前代码里的默认/预训练权重加载路径。
RAND_INIT="${RAND_INIT:-False}"

# ===== 原型/对比学习/伪标签 =====
# IS_PROTO: 是否启用 prototype 分支，这是当前脚本默认训练设置。
IS_PROTO="${IS_PROTO:-True}"
# MOM_UPDATE: 是否用动量方式更新 prototype。
MOM_UPDATE="${MOM_UPDATE:-False}"

# GLOBALEMA: 是否维护全局模型 EMA。
GLOBALEMA="${GLOBALEMA:-False}"
# PROTO_START_EPOCH: 从第几个全局 epoch/round 开始启用 prototype 相关逻辑。
PROTO_START_EPOCH="${PROTO_START_EPOCH:-1}"
# CON_LAMB / CON_LAMB_LOCAL: 全局/本地对比损失权重。
CON_LAMB="${CON_LAMB:-0.1}"
# MOMENTUM: prototype/EMA 类更新中的动量系数。
MOMENTUM="${MOMENTUM:-0.99}"
# TEMP: 对比学习 softmax 温度。
TEMP="${TEMP:-0.07}"
# EPOCH_NUM: 联邦全局训练轮数。
EPOCH_NUM="${EPOCH_NUM:-1200}"
# MAX_ANCHOR: 对比学习中每类最多采样多少 anchor，越大越慢、显存越高。
MAX_ANCHOR="${MAX_ANCHOR:-4096}"
# KMEAN_NUM: 每类 prototype 数；0/1/2 的语义取决于当前 prototype 实现。
KMEAN_NUM="${KMEAN_NUM:-2}"
# PSEUDO_LABLE: 是否使用伪标签。变量名沿用原脚本拼写，传给 Python 时是 pseudo_label。
PSEUDO_LABLE="${PSEUDO_LABLE:-True}"
# PSEUDO_LABEL_START_EPOCH: 从第几个 epoch/round 开始使用伪标签。
PSEUDO_LABEL_START_EPOCH="${PSEUDO_LABEL_START_EPOCH:-1}"
# LOCALMEM: 是否为客户端维护本地 prototype/memory。
LOCALMEM="${LOCALMEM:-True}"
CON_LAMB_LOCAL="${CON_LAMB_LOCAL:-1}"

# ===== 数据集、类别数、设备 =====
# DATASET 支持 cityscapes/camvid/ade20k/voc。
DATASET="${DATASET:-voc}" # cityscapes #ade20k #camvid
# VOC 当前默认 20 个前景类；如果你的标签包含 background，需要和数据读取逻辑一起确认。
NUM_CLS="${NUM_CLS:-20}"
# NUM_USERS: 联邦客户端总数。
NUM_USERS="${NUM_USERS:-60}"
# GPU_ID: CUDA_VISIBLE_DEVICES 风格的 GPU 编号，传给 federated_main.py。
GPU_ID="${GPU_ID-0}"

# ===== checkpoint / eval 模式 =====
# 默认启动训练；训练中的 global eval 使用快速评估，完整训练结束后再跑正常 eval。
# 只想加载已训练模型测试时：EVAL_ONLY=True bash run_voc.sh
# CHECKPOINT 为空时，eval_only 会按 federated_main.py 的默认逻辑找模型权重。
CHECKPOINT="${CHECKPOINT:-}"
# EVAL_ONLY=True: 不训练，只加载 checkpoint 跑一次 global eval。
EVAL_ONLY="${EVAL_ONLY:-False}"
# EVAL_FAST_MODE=True: 训练中使用快速评估预设；完整训练结束后的最终 eval 会回到正常精确评估。
EVAL_FAST_MODE="${EVAL_FAST_MODE:-True}"
# EVAL_TFDATA_BATCH=True: 用 tf.data 做 shape-batched eval，减少 Python/Numpy pad/stack 开销。
EVAL_TFDATA_BATCH="${EVAL_TFDATA_BATCH:-True}"
# EVAL_BUCKETS: 快速评估时把不同尺寸图片 pad 到少量固定桶，减少 TF retrace/编译次数。
# 这里默认两个桶 384x512、512x512；设为空字符串表示按真实 shape 精确评估。
if [ -z "${EVAL_BUCKETS+x}" ]; then
  if [ "${EVAL_FAST_MODE}" = "True" ] || [ "${EVAL_FAST_MODE}" = "true" ] || [ "${EVAL_FAST_MODE}" = "1" ]; then
    EVAL_BUCKETS="384x512,512x512"
  else
    EVAL_BUCKETS=""
  fi
fi
# FAST_NHWC=True: 使用 TensorFlow 原生 NHWC 前向路径，避免 NCHW 转换带来的额外开销。
FAST_NHWC="${FAST_NHWC:-True}"

# ===== 启动训练/评估 =====
# 下方参数基本是把上面的 shell 变量传给 segmentation/federated_main.py。
# 如需临时改频率，可用环境变量覆盖：
#   SAVE_FREQUENCY=10 LOCAL_TEST_FREQUENCY=9999 GLOBAL_TEST_FREQUENCY=5 bash run_voc.sh
source "$(cd "$(dirname "$0")" && pwd)/scripts/tensorflow_env.sh"
fedseg_tensorflow_prepare_for_gpu_id "${GPU_ID}"

"${FEDSEG_PYTHON[@]}" -u segmentation/federated_main.py \
--gpu="${GPU_ID}" \
--dataset=$DATASET \
--root_dir=$ROOT_DIR \
--USE_ERASE_DATA=True \
--num_classes=$NUM_CLS \
--data="train" \
--num_workers=4 \
--model="bisenetv2" \
--checkpoint="${CHECKPOINT}" \
--eval_only="${EVAL_ONLY}" \
--eval_fast_mode="${EVAL_FAST_MODE}" \
--eval_tfdata_batch="${EVAL_TFDATA_BATCH}" \
--eval_buckets="${EVAL_BUCKETS}" \
--fast_nhwc="${FAST_NHWC}" \
--lr=0.05 \
--lr_scheduler="step" \
--iid=False \
--num_users=$NUM_USERS \
--frac_num=$FRAC_NUM \
--epochs=$EPOCH_NUM \
--local_ep=$LOCAL_EP \
--local_bs=$LOCAL_BS \
--is_proto=$IS_PROTO \
--losstype=$LOSSTYPE \
--fedprox_mu=$FEDPROX_MU \
--label_online_gen=$LABEL_ONLINE_GEN \
--distill=$DISTILL \
--distill_lamb_pi=$LAMB_PI \
--distill_lamb_pa=$LAMB_PA \
--rand_init=$RAND_INIT \
--warmstep=$WARMSTEP \
--globalema=$GLOBALEMA \
--temp_dist=$TEMP_DIST \
--mixlabel=$MIXLABLE \
--proto_start_epoch=$PROTO_START_EPOCH \
--con_lamb=$CON_LAMB \
--con_lamb_local=$CON_LAMB_LOCAL \
--momentum=$MOMENTUM \
--temperature=$TEMP \
--max_anchor=$MAX_ANCHOR \
--kmean_num=$KMEAN_NUM \
--pseudo_label=$PSEUDO_LABLE \
--pseudo_label_start_epoch=$PSEUDO_LABEL_START_EPOCH \
--localmem=$LOCALMEM \
--mom_update=$MOM_UPDATE \
--save_frequency="${SAVE_FREQUENCY:-20}" \
--local_test_frequency="${LOCAL_TEST_FREQUENCY:-9999}" \
--global_test_frequency="${GLOBAL_TEST_FREQUENCY:-20}" \
--USE_WANDB="${USE_WANDB:-0}" \
--date_now=${date_now} \
2>&1 | tee -a "save/logs/log-${date_now}.txt"
