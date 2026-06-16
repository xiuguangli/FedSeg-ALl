# FedSeg-MindSpore

这个目录对应 MindSpore 版本实现仓库 [`xiuguangli/FedSeg-mindspore`](https://github.com/xiuguangli/FedSeg-mindspore)。

## 复现环境

推荐直接在当前目录使用 `uv`，环境会创建在本目录的 `.venv/` 下：

```bash
uv sync
uv run python -V
```

这套环境只保留当前代码实际需要的 MindSpore 依赖，没有额外打包 PyTorch。当前 Python 依赖固定为：

- Python `3.10`
- `mindspore==2.6.0`
- `numpy==1.26.4`

需要注意：`uv.lock` 只能固定 Python 包，MindSpore GPU 还需要宿主机提供 NVIDIA driver、CUDA runtime 和 cuDNN 动态库。当前脚本默认按 GPU 复现，推荐使用 CUDA 11.x + cuDNN 8，其中 CUDA 11.6 是 MindSpore 2.6.0 更稳妥的目标环境。

clone 后建议先检查运行时：

```bash
bash check_mindspore_runtime.sh
```

如果 CUDA 安装在非默认位置，可以显式指定：

```bash
FEDSEG_CUDA_HOME=/path/to/cuda-11.6 bash check_mindspore_runtime.sh
```

如果只是想确认 CPU 路径能跑，可以显式关闭 GPU：

```bash
GPU_ID="" bash check_mindspore_runtime.sh
```

脚本会自动探测常见 CUDA 路径，并补充 `LD_LIBRARY_PATH`。如果请求 GPU 但环境不满足，会直接报出修复提示，不再静默退回 CPU。

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

如果要指定 GPU：

```bash
GPU_ID=1 bash eval_voc.sh
```

如果要临时走 CPU：

```bash
GPU_ID="" bash eval_voc.sh
```

如果确实要绕过 `eval_voc.sh` 直接指定参数，先加载运行时 helper：

```bash
source scripts/mindspore_env.sh
fedseg_mindspore_prepare_for_gpu_id 0
uv run python -u segmentation/eval_voc.py \
  --gpu 0 \
  --dataset voc \
  --root_dir data/voc \
  --num_classes 20 \
  --data val \
  --num_workers 8 \
  --batch_size 24 \
  --model bisenetv2 \
  --checkpoints save/checkpoints/fedseg-ms-33.29.ckpt
```

## 说明

- GPU 复现前先运行 `bash check_mindspore_runtime.sh`，避免因为 CUDA/cuDNN 路径不匹配而误以为依赖没有安装好。
- 默认复现路径不包含 `wandb`，因为常用脚本默认不会启用它。
- `.venv/` 已经加入忽略规则，不会污染仓库提交。
