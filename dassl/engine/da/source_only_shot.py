import torch
import torch.nn as nn
import time
import datetime

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter


@TRAINER_REGISTRY.register()
class SourceOnlyShot(TrainerXU):
    """Baseline model for domain adaptation, which is
    trained using source data only.
    """

    def forward_backward(self, batch_x):
        input, label = self.parse_batch_train(batch_x)
        output = self.model(input)
        loss = CrossEntropyLabelSmooth(self.num_classes, self.cfg.TRAINER.SHOT.SMOOTH)(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x):
        input = batch_x['img']
        label = batch_x['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        self.num_batches = len_train_loader_x
        train_loader_x_iter = iter(self.train_loader_x)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_x = next(train_loader_x_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)

            end = time.time()


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        # TODO: F.one_hot(targets) is more simpler
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss
