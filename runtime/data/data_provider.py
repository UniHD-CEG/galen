import abc
from abc import abstractmethod

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms
from torch.utils.data import SubsetRandomSampler, Subset


class ADataProvider(metaclass=abc.ABCMeta):

    @property
    @abstractmethod
    def train_loader(self):
        pass

    @property
    @abstractmethod
    def val_loader(self):
        pass

    @property
    @abstractmethod
    def test_loader(self):
        pass

    @property
    @abstractmethod
    def sens_loader(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def batch_input_shape(self):
        pass

    @abstractmethod
    def get_random_tensor_with_input_shape(self):
        pass


class CIFAR10Provider(ADataProvider):

    def __init__(self,
                 target_device: torch.device,
                 data_dir="./data",
                 batch_size=256,
                 sensitivity_sample_count=128,
                 seed=42,
                 num_workers=16,
                 split_ratio=0.2,
                 **kwargs
                 ):
        self._batch_size = batch_size
        self._target_device = target_device
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        base_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )
        self._val_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=base_transform, download=True)
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )
        self._train_set = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_transform)
        self._test_set = torchvision.datasets.CIFAR10(data_dir, train=False, transform=base_transform)
        num_images = len(self._train_set)
        indices = list(range(num_images))
        split_idx = int(np.floor(num_images * split_ratio))
        random_gen = np.random.Generator(np.random.PCG64(seed))
        random_gen.shuffle(indices)
        train_idx, val_idx = indices[split_idx:], indices[:split_idx]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        # although the val_set is used the sens_sample ensures that train images are used for sensitivity analysis
        # train_set could not be used due to random augmentation
        sens_set = Subset(self._val_set, train_idx[:sensitivity_sample_count])

        self._train_loader = torch.utils.data.DataLoader(self._train_set, batch_size=batch_size, sampler=train_sampler,
                                                         num_workers=num_workers, pin_memory=True)
        self._valid_loader = torch.utils.data.DataLoader(self._val_set, batch_size=batch_size, sampler=valid_sampler,
                                                         num_workers=num_workers, pin_memory=True)
        self._sens_loader = torch.utils.data.DataLoader(sens_set, batch_size=batch_size, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(self._test_set, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._valid_loader

    @property
    def sens_loader(self):
        return self._sens_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def num_classes(self):
        return 10

    @property
    def batch_input_shape(self):
        # return next(iter(self.train_loader))[0].shape # causes a memory leak
        return torch.Size([self._batch_size, 3, 32, 32])

    def get_random_tensor_with_input_shape(self):
        return torch.randn(self.batch_input_shape).to(self._target_device)
