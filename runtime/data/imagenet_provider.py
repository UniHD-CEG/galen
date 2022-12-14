import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import transforms

from runtime.data.data_provider import ADataProvider

subset_of_classes = []


def _read_subset_file(subset_class_file):
    if subset_class_file:
        cls_list = []
        f = open(subset_class_file, 'r')
        for x in f:
            cls_list.append(x[:9])
        global subset_of_classes
        subset_of_classes = cls_list


class ImageNetSubsetFolder(ImageFolder):
    @staticmethod
    def make_dataset(directory: str, class_to_idx: Dict[str, int], extensions: Optional[Tuple[str, ...]] = None,
                     is_valid_file: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, int]]:
        duplicate_class_to_idx = copy.deepcopy(class_to_idx)
        if len(subset_of_classes) > 0:
            for clazz in class_to_idx.keys():
                if clazz not in subset_of_classes:
                    del duplicate_class_to_idx[clazz]

        return make_dataset(directory, duplicate_class_to_idx, extensions, is_valid_file)


class ImageNetDataProvider(ADataProvider):
    def __init__(self,
                 target_device: torch.device,
                 data_dir,
                 batch_size=256,
                 sensitivity_sample_count=128,
                 seed=42,
                 num_workers=16,
                 train_size=None,
                 val_size=None,
                 subset_classes_file=None,
                 **kwargs
                 ):
        self._batch_size = batch_size
        self._target_device = target_device
        self.random_gen = np.random.Generator(np.random.PCG64(seed))
        _read_subset_file(subset_classes_file)
        root_path = Path(data_dir)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_path = root_path / "train"
        train_set = ImageNetSubsetFolder(str(train_path),
                                         transform=transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ]))
        self._num_classes = len(train_set.classes)

        val_path = root_path / "val"
        val_set = ImageNetSubsetFolder(str(val_path),
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))

        if not train_size:
            train_size = len(train_set)
        else:
            train_size = int(train_size)
        train_sampler = self.create_subset_sampler(len(train_set), train_size)
        self._train_loader = torch.utils.data.DataLoader(train_set,
                                                         batch_size=self._batch_size,
                                                         sampler=train_sampler,
                                                         num_workers=num_workers,
                                                         pin_memory=True)
        if not val_size:
            val_size = len(val_set)
        else:
            val_size = int(val_size)
        val_sampler = self.create_subset_sampler(len(val_set), val_size)
        self._val_loader = torch.utils.data.DataLoader(val_set,
                                                       batch_size=self._batch_size,
                                                       sampler=val_sampler,
                                                       num_workers=num_workers,
                                                       pin_memory=True)

        sens_indices = self.sample_indices(len(train_set), sensitivity_sample_count)
        sens_set = Subset(train_set, sens_indices)
        self._sens_loader = torch.utils.data.DataLoader(sens_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        pin_memory=True)

    def create_subset_sampler(self, actual_size, target_size, sampler=SubsetRandomSampler):
        indices = self.sample_indices(actual_size, target_size)
        train_sampler = sampler(indices)
        return train_sampler

    def sample_indices(self, actual_size, target_size):
        indices = np.arange(actual_size)
        self.random_gen.shuffle(indices)
        indices = indices[:target_size]
        return indices

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def test_loader(self):
        return self._val_loader

    @property
    def sens_loader(self):
        return self._sens_loader

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def batch_input_shape(self):
        return torch.Size([self._batch_size, 3, 224, 224])

    def get_random_tensor_with_input_shape(self):
        return torch.randn(self.batch_input_shape).to(self._target_device)
