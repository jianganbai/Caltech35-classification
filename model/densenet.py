import torch.nn as nn
import torchvision.models as models


class dense(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.dense = models.densenet201(pretrained=True)
        for param in self.dense.parameters():
            param.requires_grad = False
        self.dense.classifier = nn.Linear(1920, 64)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num)
        )

        self.featout = False

    def set_featout(self, property):
        self.featout = property

    def forward(self, data):
        feat = self.dense(data)
        x = self.fc(feat)
        if self.featout:
            return x, feat
        else:
            return x


def densenet(class_num=35):
    net = dense(class_num)
    # print(net)
    return net


if __name__ == '__main__':
    densenet(35)
