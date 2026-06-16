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

建议始终先进入对应后端目录，再执行脚本。

### 1. PyTorch

```bash
cd FedSeg-Torch
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

VOC 评估示例：

```bash
cd FedSeg-Torch
bash eval_voc.sh save/checkpoints/your_checkpoint.pth
```

## 2. TensorFlow 2

```bash
cd FedSeg-Tensorflow2
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

VOC 评估示例：

```bash
cd FedSeg-Tensorflow2
bash eval_voc.sh save/checkpoints/FedSeg.weights.h5
```

TensorFlow2 目录下还带了测试，可以按需执行：

```bash
cd FedSeg-Tensorflow2
pytest test -q
```

`run_voc.sh` 支持不少环境变量覆盖，例如：

```bash
cd FedSeg-Tensorflow2
GPU_ID=0 ROOT_DIR=data/voc EVAL_ONLY=True bash run_voc.sh
```

### 3. MindSpore

```bash
cd FedSeg-MindSpore
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

评估示例：

```bash
cd FedSeg-MindSpore
bash eval_voc.sh save/checkpoints/your_checkpoint.ckpt
```

MindSpore 目录里的说明建议使用单独环境，例如：

```bash
micromamba run -n fedseg-mindspore python -V
```

### 4. Paddle

```bash
cd FedSeg_Paddle
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
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
