import torch
import torch.nn as nn
import os.path as osp

from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.engine.trainer import SimpleNet
from dassl.utils import count_num_param, load_checkpoint


@TRAINER_REGISTRY.register()
class MSFDA(TrainerXU):
    """Baseline model for domain adaptation, which is
    trained using source data only.
    """

    def build_model(self):
        cfg = self.cfg

        print('Building multiple source models')
        self.models = nn.ModuleList(
            [
                SimpleNet(
                    cfg, cfg.MODEL, self.num_classes, cfg.MODEL.CLASSIFIER.TYPE
                ) for _ in range(self.dm.num_source_domains)
            ]
        )
        self.models.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.models)))
        self.register_model('models', self.models)

    def load_model(self, directory, epoch=None):
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for i, domain in enumerate(self.cfg.DATASET.SOURCE_DOMAINS):
            model_path = osp.join(directory, domain, 'model', model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to the model of {} '
                'from "{}" (epoch = {})'.format(domain, model_path, epoch)
            )
            self.models[i].load_state_dict(state_dict)

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

    def model_inference(self, input):
        p = 0
        for model_i in self.models:
            z = model_i(input)
            p += F.softmax(z, 1)
        p = p / len(self.models)
        return p
