import random
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

# Folder names for train and test sets
MNIST = {'train': 'train_images', 'test': 'test_images'}
MNIST_M = {'train': 'train_images', 'test': 'test_images'}
SVHN = {'train': 'train_images', 'test': 'test_images'}
SYN = {'train': 'train_images', 'test': 'test_images'}
USPS = {'train': 'train_images', 'test': 'test_images'}


def read_image_list(im_dir, n_max=None, n_repeat=None):
    items = []

    for imname in listdir_nohidden(im_dir):
        imname_noext = osp.splitext(imname)[0]
        label = int(imname_noext.split('_')[1])
        impath = osp.join(im_dir, imname)
        items.append((impath, label))

    if n_max is not None:
        items = random.sample(items, n_max)

    if n_repeat is not None:
        items *= n_repeat

    return items


def load_mnist(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST[split])
    return read_image_list(data_dir)


def load_mnist1(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST[split])
    return read_image_list(data_dir)


def load_mnist_m(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST_M[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_svhn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SVHN[split])
    return read_image_list(data_dir)


def load_syn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SYN[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_usps(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, USPS[split])
    return read_image_list(data_dir)


@DATASET_REGISTRY.register()
class Digit(DatasetBase):
    """Three digit datasets.

    It contains:
        - MNIST: hand-written digits. 28x28. Train:60000, Test:10000.
        - SVHN: street view house number. 32x32. Train:73257, Test:26032.
        - USPS: hand-written digits. 28x28. Train:7438, Test:1860.

    """
    dataset_dir = 'digit'
    domains = ['mnist', 'svhn', 'usps', 'mnist1']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='train')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, split='train'):
        items = []

        for domain, dname in enumerate(input_domains):
            func = 'load_' + dname
            domain_dir = osp.join(self.dataset_dir, dname)
            items_d = eval(func)(domain_dir, split=split)

            for impath, label in items_d:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items
