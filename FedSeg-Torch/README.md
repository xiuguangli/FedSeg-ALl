# FedSeg-Torch

这个目录对应原始 PyTorch 实现仓库 [`lightas/FedSeg`](https://github.com/lightas/FedSeg)。

## 复现环境

推荐直接在当前目录使用 `uv`，环境会创建在本目录的 `.venv/` 下：

```bash
uv sync
```

不需要手动激活环境时，可以直接这样运行：

```bash
uv run python -V
bash check_torch_runtime.sh
```

默认依赖已经固定在 `pyproject.toml` 和 `uv.lock` 里，适合 clone 后直接复现。Linux 下会从 PyTorch 官方 `cu128` 源安装 CUDA 版 wheel：

- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`

宿主机仍然需要有可用的 NVIDIA driver。clone 后建议先运行 `bash check_torch_runtime.sh`，确认 PyTorch 能看到 GPU。

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

如果只是想临时走 CPU：

```bash
GPU_ID="" bash eval_voc.sh
```

如果确实要绕过 `eval_voc.sh` 直接指定参数，先加载运行时 helper：

```bash
source scripts/torch_env.sh
fedseg_torch_prepare_for_gpu_id 0
uv run python -u segmentation/eval_voc.py \
  --gpu 0 \
  --dataset voc \
  --root_dir data/voc \
  --num_classes 20 \
  --data val \
  --num_workers 2 \
  --batch_size 1 \
  --model bisenetv2 \
  --checkpoints save/checkpoints/FedSeg1.pth
```

## 说明

- 默认按 GPU 复现；如果请求 GPU 但 PyTorch 看不到 GPU，会直接报错而不是静默退回 CPU。
- 默认脚本里 `USE_WANDB=0`，因此复现环境没有把 `wandb` 作为基础依赖。
- `.venv/` 已经加入忽略规则，不会污染仓库提交。
