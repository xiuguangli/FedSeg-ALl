# FedSeg_Paddle

这个目录对应 Paddle 版本实现仓库 [`18079070189/FedSeg_Paddle`](https://github.com/18079070189/FedSeg_Paddle.git)。

## 复现环境

推荐直接在当前目录使用 `uv`，环境会创建在本目录的 `.venv/` 下：

```bash
uv sync
```

`paddlepaddle-gpu==3.3.0` 不在默认 PyPI 上，所以这里额外配置了 Paddle 官方 `cu129` wheel 源。

## 实际入口

这个目录里同时保留了 `segmentation/` 和 `paddle_segmentation/` 两套历史代码。

当前真正的 Paddle 版本实现位于：

```text
paddle_segmentation/
```

复现和验证时建议优先使用这一套。

## 数据目录

默认数据仍然建议放在当前目录下的 `data/`。

## 常用命令

如果你后面要继续整理 Paddle 版，建议优先从 `paddle_segmentation/` 下的脚本和模块开始。

当前验证采用的是 `paddlepaddle-gpu==3.3.0`。实测这份代码在 2.5/2.6 系列会遇到初始化 API 不兼容，3.3.0 可以在不改源码的前提下完成模型前向。

最小前向验证可以直接在当前目录运行：

```bash
uv run python -c "import paddle; from types import SimpleNamespace; from paddle_segmentation.myseg.bisenetv2 import BiSeNetV2; model = BiSeNetV2(SimpleNamespace(proj_dim=256, rand_init=True), 20, aux_mode='eval'); x = paddle.randn([1, 3, 512, 512]); y = model(x)[0]; print(tuple(y.shape))"
```

## 说明

- 这套环境默认不启用 `wandb`。
- `.venv/` 已经加入忽略规则，不会污染仓库提交。
