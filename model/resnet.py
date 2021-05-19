import torch.nn as nn
from torchvision import models


def resnet(class_num):
    net = models.resnet34(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(512, class_num)
    # print(net)
    return net


if __name__ == '__main__':
    resnet(35)
