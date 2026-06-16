# FedSeg MindSpore

MindSpore-only refactor of FedSeg for semantic segmentation.

## Environment

Use the `fedseg-mindspore` micromamba environment for training and evaluation:

```bash
micromamba run -n fedseg-mindspore python -V
```

## Dataset Layout

Put the prepared datasets under `data/`, for example:

- `data/voc`
- `data/cityscapes_split_erase19`
- `data/camvid_erase_11C1`

## Training

```bash
bash run_voc.sh
bash run_city.sh
bash run_camvid.sh
bash run_ade20k.sh
```

## Evaluation

MindSpore checkpoints use the `.ckpt` suffix.

```bash
bash eval_voc.sh save/checkpoints/your_checkpoint.ckpt
bash eval.sh saved.ckpt
```
