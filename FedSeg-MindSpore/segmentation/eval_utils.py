import numpy as np
from tqdm import tqdm


def _make_iterator(data_loader):
    if hasattr(data_loader, "create_tuple_iterator"):
        return data_loader.create_tuple_iterator(output_numpy=False, num_epochs=1)
    return iter(data_loader)


def _dataset_size(data_loader):
    if hasattr(data_loader, "get_dataset_size"):
        return data_loader.get_dataset_size()
    return None


def evaluate(model, data_loader, num_classes):
    model.set_train(False)
    confmat = ConfusionMatrix(num_classes)
    eval_loader = tqdm(
        _make_iterator(data_loader),
        total=_dataset_size(data_loader),
        desc="Test",
        leave=False,
        dynamic_ncols=True,
    )
    original_aux_mode = getattr(model, "aux_mode", None)
    for image, target in eval_loader:
        if original_aux_mode is not None:
            model.aux_mode = "eval"
        output = model(image)[0]
        if original_aux_mode is not None:
            model.aux_mode = original_aux_mode
        confmat.update(target.reshape(-1), output.argmax(axis=1).reshape(-1))
    confmat.compute()
    return confmat


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.acc_global = 0.0
        self.iou_mean = 0.0
        self.acc = np.zeros((num_classes,), dtype=np.float32)
        self.iu = np.zeros((num_classes,), dtype=np.float32)

    def update(self, target, prediction):
        target_np = target.asnumpy()
        prediction_np = prediction.asnumpy()
        self.update_numpy(target_np, prediction_np)

    def update_numpy(self, target_np, prediction_np):
        target_np = target_np.astype(np.int64, copy=False)
        prediction_np = prediction_np.astype(np.int64, copy=False)
        mask = (target_np >= 0) & (target_np < self.num_classes)
        indices = self.num_classes * target_np[mask] + prediction_np[mask]
        self.mat += np.bincount(indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def reset(self):
        self.mat.fill(0)

    def compute(self):
        hist = self.mat.astype(np.float32)
        total = hist.sum()
        if total <= 0:
            self.acc_global = 0.0
            self.iou_mean = 0.0
            self.acc.fill(0.0)
            self.iu.fill(0.0)
            return

        diag = np.diag(hist)
        self.acc_global = float(diag.sum() / total * 100.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.acc = diag / hist.sum(axis=1)
            denom = hist.sum(axis=1) + hist.sum(axis=0) - diag
            self.iu = diag / denom
        valid_iu = self.iu[~np.isnan(self.iu)]
        self.iou_mean = float(valid_iu.mean() * 100.0) if valid_iu.size else 0.0

    def __str__(self):
        self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            self.acc_global,
            ["{:.1f}".format(value) for value in (self.acc * 100.0).tolist()],
            ["{:.1f}".format(value) for value in (self.iu * 100.0).tolist()],
            self.iou_mean,
        )
