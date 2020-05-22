import torch
import torch.nn as nn
import numpy as np
import copy
import datetime
import time
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from dassl.utils import check_isfile, open_specified_layers
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_specified_pretrained_weights
from dassl.metrics import compute_accuracy

__all__ = ['KD']


def Entropy(input_):
    # bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(
        self, num_classes, epsilon=0.1, use_gpu=True, size_average=True
    ):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        # TODO: F.one_hot(targets) is more simpler
        targets = torch.zeros(log_probs.size()
                              ).scatter_(1,
                                         targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (
            1 - self.epsilon
        ) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


@TRAINER_REGISTRY.register()
class KD(TrainerXU):
    """Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

    https://arxiv.org/abs/2002.08546
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fix_model = SimpleNet(
            cfg, cfg.MODEL, self.num_classes, cfg.MODEL.CLASSIFIER.TYPE
        )
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        load_pretrained_weights(
            self.fix_model,
            'output/shot_source/office31/a2w/model/model.pth.tar-30'
        )
        self.fix_model.to(self.device)
        self.fix_model.eval()

        self.open_layers = ['backbone']
        if isinstance(self.model.head, nn.Module):
            self.open_layers.append('head')
        open_specified_layers(self.model, self.open_layers)
        # self.model.train()
        # TODO: filter out the parameters with 'requires_grad == False' from optimizer

    # def check_cfg(self, cfg):
    #     assert check_isfile(
    #         cfg.MODEL.INIT_WEIGHTS
    #     ), 'The weights of source model must be provided'

    def parse_batch_train(self, batch_u):
        input = batch_u['img']
        label = batch_u['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def forward_backward(self, batch_u):
        input_u, label = self.parse_batch_train(batch_u)
        with torch.no_grad():
            # _, features = self.model(input_u, return_feature=True)
            features = self.netH(self.netB(input_u))
            pred = self.obtain_label(features, self.center)

        outputs = self.model(input_u)
        classifier_loss = CrossEntropyLabelSmooth(self.num_classes,
                                                  0)(outputs, pred)
        softmax_out = nn.Softmax(dim=1)(outputs)
        im_loss = torch.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        im_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

        with torch.no_grad():
            outputs_fix = self.fix_model(input_u)

        T = 1.0
        kd_loss = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            F.softmax(outputs_fix / T, dim=1),
            reduction='batchmean'
        ) * T * T

        loss = im_loss + 0.3*classifier_loss + kd_loss

        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(outputs, label)[0].item()
        }

        return loss_summary

    def obtain_label(self, features_target, center):
        features_target = torch.cat(
            (features_target, torch.ones(features_target.size(0), 1).cuda()), 1
        )
        fea = features_target.float().detach().cpu().numpy()
        center = center.float().detach().cpu().numpy()
        dis = cdist(fea, center, 'cosine') + 1
        pred = np.argmin(dis, axis=1)
        pred = torch.from_numpy(pred).cuda()
        return pred

    def obtain_center(self):
        loader = self.train_loader_u
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()
                inputs = data['img']
                labels = data['label']
                inputs = inputs.to(self.device)
                # outputs, feas = self.model(inputs, return_feature=True)
                feas = self.netH(self.netB(inputs))
                outputs = self.model.classifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0
                    )
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label
                             ).item() / float(all_label.size()[0])

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        # initial center
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        center = torch.from_numpy(initc).cuda()

        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(
            accuracy * 100, acc * 100
        )
        print(log_str + '\n')
        return center

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_u = len(self.train_loader_u)
        self.num_batches = len_train_loader_u
        train_loader_u_iter = iter(self.train_loader_u)

        # self.fix_model.load_state_dict(self.model.state_dict())
        # self.fix_model.eval()
        self.netB = copy.deepcopy(self.model.backbone)
        self.netH = copy.deepcopy(self.model.head)
        self.netB.eval()
        self.netH.eval()
        self.center = self.obtain_center()

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_u = next(train_loader_u_iter)
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_u)
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
