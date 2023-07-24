import torch
import torch.hub
from torch import nn
from pathlib import Path
import torchvision

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        bsc_cin = 32
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=bsc_cin, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=bsc_cin, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(in_features=32768, out_features=10)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = torch.flatten(res, 1)
        return self.linear(res)


def test_model():
    return TestModel()


def resnet18_cifar():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

def mobile_net_pretrained():
    model = torchvision.models.mobilenet_v2(pretrained=True)
    num_classes = 10
    model.classifier[1] = torch.nn.Linear(1280, num_classes)
    return model

def resnet18_pretrained():
    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model

provider = {
    "test_model": test_model,
    "resnet18_cifar": resnet18_cifar,
    "mobile_net_pretrained": mobile_net_pretrained,
    "resnet18_pretrained": resnet18_pretrained
}


def load_model(select_str, num_classes, checkpoint_path=None):
    if "@" in select_str:
        # model on torch hub
        name, repo = select_str.split("@")
        model = torch.hub.load(repo, name, pretrained=True, num_classes=num_classes)
    elif "/" in select_str:
        # local path to pretrained model containing architecture definition
        model = torch.load(select_str)
        name = Path(select_str).stem
    else:
        # predefined architecture definition
        model = provider[select_str]()
        name = select_str

    if checkpoint_path is not None:
        model.load_state_dict(load_checkpoint(checkpoint_path))
    return model, name


def load_checkpoint(checkpoint_path):
    if checkpoint_path.endswith('.lightning.ckpt'):
        state_dict = torch.load(checkpoint_path)['state_dict']
        return {key[6:]: weight for key, weight in state_dict.items()}
    if checkpoint_path.startswith('pretrained_checkpoints'):
        return torch.load(checkpoint_path)['model_state_dict']
    return torch.load(checkpoint_path)
