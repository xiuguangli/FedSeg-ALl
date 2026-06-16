import copy
import time

import myseg.bisenet_utils
import paddle
from eval_utils import evaluate
from myseg.bisenet_utils import BackCELoss, OhemCELoss
from myseg.magic import MultiEpochsDataLoader


class DatasetSplit(paddle.io.Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach().astype(dtype="float32"), label.clone().detach()


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:]
        idxs_test = idxs[: int(0.5 * len(idxs))]
        trainloader = MultiEpochsDataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args.local_bs,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        testloader = paddle.io.DataLoader(
            dataset=DatasetSplit(dataset, idxs_test),
            batch_size=2,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        return trainloader, testloader

    @paddle.no_grad()
    def get_protos(self, model, global_round):
        args = self.args
        model.eval()
        tmp_ = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)
            if args.model == "bisenetv2":
                logits, feat_head, *logits_aux = model(images)
            _, _, h, w = tuple(feat_head.shape)
            feat_head = feat_head.unsqueeze(axis=1)
            labels = labels.unsqueeze(axis=1)
            labels = paddle.nn.functional.interpolate(
                x=labels.astype(dtype="float32"), size=(h, w), mode="nearest"
            )
            labels = labels.unsqueeze(axis=1)
            class_ = (
                paddle.arange(end=args.num_classes)
                .to(labels.place)
                .unsqueeze(axis=0)
                .unsqueeze(axis=-1)
                .unsqueeze(axis=-1)
                .unsqueeze(axis=-1)
            )
            weight_ = class_ == labels
            weight_ = weight_ / (
                weight_.sum(axis=0, keepdim=True)
                .sum(axis=3, keepdim=True)
                .sum(axis=4, keepdim=True)
                + 1e-05
            )
            out = weight_ * feat_head
            out = out.sum(axis=0).sum(axis=-1).sum(axis=-1)
            tmp_.append(out)
        tmp_ = sum(tmp_) / len(tmp_)
        return tmp_

    def update_weights(self, model, global_round, prototypes=None):
        if prototypes is not None:
            prototypes = prototypes.transpose(perm=[1, 0])
        model.train()
        epoch_loss = []
        args = self.args
        if args.is_proto and not args.label_online_gen:
            model_record = copy.deepcopy(model)
            model_record.eval()
        if args.model == "bisenetv2":
            optimizer = myseg.bisenet_utils.set_optimizer(model, args)
            if args.losstype == "ohem":
                criteria_pre = OhemCELoss(0.7)
                criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
            elif args.losstype == "ce":
                criteria_pre = paddle.nn.CrossEntropyLoss(
                    ignore_index=255, reduction="mean"
                )
                criteria_aux = [
                    paddle.nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
                    for _ in range(4)
                ]
            elif args.losstype == "back":
                criteria_pre = BackCELoss(args)
                criteria_aux = [BackCELoss(args) for _ in range(4)]
            else:
                raise ValueError("loss type is not defined")
        else:
            exit("Error: unrecognized model")
        tmp_lr = paddle.optimizer.lr.LambdaDecay(
            lr_lambda=lambda x: 1 if global_round < 1000 else 0.1,
            learning_rate=optimizer.get_lr(),
        )
        optimizer.set_lr_scheduler(tmp_lr)
        tmp_lr = paddle.optimizer.lr.LambdaDecay(
            lr_lambda=lambda x: (
                1 - x / (len(self.trainloader) * max(1, args.local_ep))
            )
            ** 0.9,
            learning_rate=optimizer.get_lr(),
        )
        scheduler_dict = {"step": tmp_lr, "poly": tmp_lr}
        lr_scheduler = scheduler_dict[args.lr_scheduler]
        start_time = time.time()
        for iter in range(args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if args.model == "bisenetv2":
                    logits, feat_head, *logits_aux = model(images)
                    if args.is_proto:
                        if not args.label_online_gen:
                            with paddle.no_grad():
                                _, feat_head, *logits_aux_tmp = model_record(images)
                        _, h, w = tuple(labels.shape)
                        feat_head = feat_head.transpose(perm=[0, 2, 3, 1])
                        feat_head = paddle.nn.functional.normalize(x=feat_head, p=-1)
                        new_label = paddle.matmul(x=feat_head, y=prototypes)
                        new_label = new_label.transpose(perm=[0, 3, 1, 2]).detach()
                        new_label = paddle.nn.functional.interpolate(
                            x=new_label, size=(h, w), mode="nearest"
                        )
                        new_label = paddle.argmax(x=new_label, axis=1)
                        labels_ = new_label.astype(dtype="int64")
                    else:
                        labels_ = labels
                    loss_pre = criteria_pre(logits, labels_)
                    loss_aux = [
                        crit(lgt, labels_)
                        for crit, lgt in zip(criteria_aux, logits_aux)
                    ]
                    loss = loss_pre + sum(loss_aux)
                else:
                    exit("Error: unrecognized model")
                loss.backward()
                optimizer.step()
                optimizer.clear_gradients(set_to_zero=False)
                batch_loss.append(loss.item())
                print(
                    "Local Epoch: {}, batch_idx: {}, lr: {:.3e}".format(
                        iter, batch_idx, lr_scheduler.get_lr()[0]
                    )
                )
                lr_scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if args.verbose:
                string = "| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}".format(
                    global_round, iter + 1, len(self.trainloader.dataset), loss.item()
                )
                print(string)
        strings = [
            "| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}".format(
                global_round, args.local_ep, len(self.trainloader.dataset), loss.item()
            )
        ]
        print("".join(strings))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """Returns the inference accuracy and loss."""
        confmat = evaluate(model, self.testloader, self.device, self.args.num_classes)
        return confmat.acc_global, confmat.iou_mean, str(confmat)


def test_inference(args, model, testloader):
    """Returns the test accuracy and loss."""
    device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)
