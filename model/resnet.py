import torch.nn as nn
from torchvision import models


class res(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.res = models.resnet34(pretrained=True)
        for param in self.res.parameters():
            param.requires_grad = False
        self.res.fc = nn.Linear(512, 64)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num)
        )

        self.featout = False

    def set_featout(self, property):
        self.featout = property

    def forward(self, data):
        feat = self.res(data)
        x = self.classifier(feat)
        if self.featout:
            return x, feat
        else:
            return x


def resnet(class_num=35):
    net = res(class_num)
    # print(net)
    return net


if __name__ == '__main__':
    resnet(35)
