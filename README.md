# FedSeg 多后端仓库说明

这个仓库可以看作原始项目 [`lightas/FedSeg`](https://github.com/lightas/FedSeg) 的 4 个版本实现集合，围绕同一篇工作 `FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation` 整理了不同后端的代码。

当前仓库里主要包含这 4 个实现版本：

- `FedSeg-Torch`：PyTorch 版本
- `FedSeg-Tensorflow2`：TensorFlow 2 版本
- `FedSeg-MindSpore`：MindSpore 版本
- `FedSeg_Paddle`：Paddle 版本

每个子项目都保留了各自的训练脚本、评估脚本和模型实现；其中 TensorFlow2 目录额外带了一组测试文件，MindSpore 目录的 README 和脚本也相对更贴近当前整理后的状态。

## 目录结构

```text
fed-seg/
├── FedSeg-Torch/
├── FedSeg-Tensorflow2/
├── FedSeg-MindSpore/
├── FedSeg_Paddle/
├── .gitignore
└── README.md
```

几个子项目的内部结构大体一致：

- `segmentation/`：核心训练、评估、联邦聚合和数据处理代码
- `run_*.sh`：训练入口脚本
- `eval*.sh`：评估入口脚本
- `data/`：数据集根目录
- `logs/`：运行日志
- `save/`：模型权重、评估结果和中间产物

## 数据准备

仓库中的脚本默认使用下面这些数据目录名：

- `data/voc`
- `data/cityscapes_split_erase19`
- `data/camvid_erase_11C1`
- `data/ade20k_erase_150C1`

也就是说，通常应该把数据放到各子项目目录下，例如：

```text
FedSeg-Torch/data/voc
FedSeg-Tensorflow2/data/cityscapes_split_erase19
FedSeg-MindSpore/data/camvid_erase_11C1
```

已有 README 中提供的数据链接如下：

- Cityscapes
  - https://pan.baidu.com/s/15D-Eq0om1DFpsKFeuBB3sg  密码: `9cbm`
  - https://pan.baidu.com/s/1AF9HKEF9fpulBOds3p_ZPQ  密码: `bvsa`
- CamVid
  - https://pan.baidu.com/s/1suB1zIQTNt02fqJBwdxMSA  密码: `l343`
  - https://pan.baidu.com/s/1WksbT44mrylLptN4wKoxqA  密码: `l610`
- Pascal VOC
  - https://pan.baidu.com/s/1c03gu0SIUA62FC4403f9GQ  密码: `3d60`
- ADE20K
  - https://pan.baidu.com/s/13ypIWZFCa58oZT7cA3KWNA  密码: `tps2`

## 快速开始

建议始终先进入对应后端目录，再执行环境配置和运行命令。四个子项目都已经用 `uv` 固定了依赖版本，环境会创建在各自目录的 `.venv/` 下，不会互相污染。

如果机器上还没有 `uv`，可以先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

下面的训练脚本建议使用 `uv run bash ...` 启动，因为脚本内部调用的是 `python`；这样可以确保实际使用的是当前子项目 `.venv/` 里的解释器和依赖。

### 1. PyTorch 版本

对应目录：`FedSeg-Torch`

环境配置：

```bash
cd FedSeg-Torch
uv sync
uv run python -V
```

当前环境主要固定为：

- Python `3.11`
- `torch==2.8.0`
- `torchvision==0.23.0`

训练示例：

```bash
uv run bash run_voc.sh
uv run bash run_city.sh
uv run bash run_camvid.sh
uv run bash run_ade20k.sh
```

VOC 评估示例：

```bash
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

### 2. TensorFlow 2 版本

对应目录：`FedSeg-Tensorflow2`

环境配置：

```bash
cd FedSeg-Tensorflow2
uv sync
uv run python -V
bash check_tensorflow_runtime.sh
```

当前环境主要固定为：

- Python `3.12`
- `tensorflow[and-cuda]==2.20.0`
- `keras==3.12.0`
- CPU 版 `torch==2.6.0` / `torchvision==0.21.0`

TensorFlow GPU 依赖通过官方 `and-cuda` extra 固定在 `uv.lock` 里，会安装对应的 `nvidia-*-cu12` wheel。宿主机仍然需要有可用的 NVIDIA driver；clone 后先跑 `bash check_tensorflow_runtime.sh`，确认 TensorFlow 能看到 GPU。这里保留 CPU 版 PyTorch 依赖，是因为 TensorFlow2 实现里有和 PyTorch checkpoint 对齐相关的代码。

训练示例：

```bash
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

只评估 VOC：

```bash
bash eval_voc.sh
```

如果只想临时走 CPU：

```bash
GPU_ID="" bash eval_voc.sh
```

如果确实要直接指定参数运行，先加载 TensorFlow 运行时 helper：

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

轻量测试：

```bash
uv run pytest test/test_eval_voc_smoke.py -q
```

`run_voc.sh` 支持用环境变量覆盖参数，例如：

```bash
GPU_ID=0 ROOT_DIR=data/voc EVAL_ONLY=True bash run_voc.sh
```

### 3. MindSpore 版本

对应目录：`FedSeg-MindSpore`

环境配置：

```bash
cd FedSeg-MindSpore
uv sync
uv run python -V
bash check_mindspore_runtime.sh
```

当前环境主要固定为：

- Python `3.10`
- `mindspore==2.6.0`
- `numpy==1.26.4`

注意：`uv.lock` 固定的是 Python 依赖，MindSpore GPU 还需要宿主机提供 NVIDIA driver、CUDA runtime 和 cuDNN 动态库。这个版本建议使用 CUDA 11.x + cuDNN 8，推荐 CUDA 11.6。脚本会自动探测常见 CUDA 路径；如果你的 CUDA 安装在自定义位置，可以这样指定：

```bash
FEDSEG_CUDA_HOME=/path/to/cuda-11.6 bash check_mindspore_runtime.sh
```

训练示例：

```bash
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

只评估 VOC：

```bash
bash eval_voc.sh
```

如果只想临时走 CPU，可以显式关闭 GPU：

```bash
GPU_ID="" bash eval_voc.sh
```

也可以直接指定参数运行：

```bash
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

### 4. Paddle 版本

对应目录：`FedSeg_Paddle`

环境配置：

```bash
cd FedSeg_Paddle
uv sync
uv run python -V
```

当前环境主要固定为：

- Python `3.10`
- `paddlepaddle-gpu==3.3.0`
- Paddle 官方 `cu129` wheel 源

这个目录里同时保留了 `segmentation/` 和 `paddle_segmentation/` 两套历史代码；当前真正的 Paddle 实现建议优先看 `paddle_segmentation/`。

最小前向验证：

```bash
uv run python -c "import paddle; from types import SimpleNamespace; from paddle_segmentation.myseg.bisenetv2 import BiSeNetV2; model = BiSeNetV2(SimpleNamespace(proj_dim=256, rand_init=True), 20, aux_mode='eval'); x = paddle.randn([1, 3, 512, 512]); y = model(x)[0]; print(tuple(y.shape))"
```

历史训练脚本仍然保留：

```bash
uv run bash run_city.sh
uv run bash run_camvid.sh
uv run bash run_ade20k.sh
```

如果运行 VOC，请先检查 `run_voc.sh` 里的 `ROOT_DIR`。当前脚本里默认还是较早的 `../voc` 风格路径，通常需要按你本地目录改成类似 `data/voc` 的形式。

## 常用输出位置

训练或评估后，常见产物一般会出现在这些目录：

- `logs/federated_main/`：训练日志
- `logs/eval/` 或 `logs/eval_voc/`：评估日志
- `logs/profile_hot_clients/`：一些 profile / probe / 对齐分析日志
- `save/checkpoints/`：模型权重
- `save/logs/`：脚本 `tee` 出来的日志文件

## 使用建议

- 不同后端的脚本默认 GPU 编号不完全一致，运行前最好先看一下 `run_*.sh` 里的 `--gpu` 或 `GPU_ID`。
- 大部分脚本直接把数据集路径写在脚本顶部；如果你的目录布局不同，优先修改 `ROOT_DIR`。
- 这个聚合仓库里包含多个子目录，各子目录也保留了自己的实现和历史痕迹；做改动时建议只在目标后端目录内工作。
- 数据集、日志、缓存和权重文件已经适合通过 `.gitignore` 忽略，不建议直接提交这些运行产物。

## 参考文献

```bibtex
@inproceedings{miao2023fedseg,
  title={FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation},
  author={Miao, Jiaxu and Yang, Zongxin and Fan, Leilei and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8042--8052},
  year={2023}
}
```
