# FedSeg-Tensorflow2

这个目录对应 TensorFlow 2 版本实现仓库 [`xiuguangli/FedSeg-Tensorflow2`](https://github.com/xiuguangli/FedSeg-Tensorflow2)。

## 复现环境

推荐直接在当前目录使用 `uv`，环境会创建在本目录的 `.venv/` 下：

```bash
uv sync
uv run python -V
bash check_tensorflow_runtime.sh
```

当前配置默认采用：

- `tensorflow[and-cuda]==2.20.0`
- `keras==3.12.0`
- CPU 版 `torch/torchvision`

TensorFlow GPU 依赖通过官方 `and-cuda` extra 固定在 `uv.lock` 里，会安装对应的 `nvidia-*-cu12` wheel。宿主机仍然需要有可用的 NVIDIA driver，clone 后建议先运行 `bash check_tensorflow_runtime.sh`，确认 TensorFlow 能看到 GPU。

如果只是想临时走 CPU：

```bash
GPU_ID="" bash check_tensorflow_runtime.sh
```

TensorFlow 目录里保留了和 PyTorch checkpoint 对齐相关的逻辑，所以仍然保留了 CPU 版 `torch/torchvision`。

## 数据目录

默认脚本使用当前目录下的 `data/`，例如：

```text
data/voc
data/cityscapes_split_erase19
data/camvid_erase_11C1
data/ade20k_erase_150C1
```

## 常用命令

训练：

```bash
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

VOC 评估：

```bash
bash eval_voc.sh
```

如果确实要绕过 `eval_voc.sh` 直接指定参数，先加载运行时 helper：

```bash
source scripts/tensorflow_env.sh
fedseg_tensorflow_prepare_for_gpu_id 0
uv run python -u segmentation/eval_voc.py \
  --gpu 0 \
  --dataset voc \
  --root ./ \
  --root_dir data/voc \
  --USE_ERASE_DATA True \
  --num_classes 20 \
  --data val \
  --num_workers 4 \
  --eval_bs 8 \
  --eval_tfdata_batch True \
  --model bisenetv2 \
  --checkpoints save/checkpoints/fedseg-tf.weights.h5
```

轻量 smoke test：

```bash
uv run pytest test/test_eval_voc_smoke.py -q
```

## 说明

- 默认按 GPU 复现；如果请求 GPU 但 TensorFlow 看不到 GPU，会直接报错而不是静默退回 CPU。
- CPU 模式可以用 `GPU_ID="" bash eval_voc.sh` 显式启用。
- 默认脚本里 `USE_WANDB=0`，因此基础环境没有把 `wandb` 作为默认依赖。
- `.venv/` 已经加入忽略规则，不会污染仓库提交。
