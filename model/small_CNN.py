import torch.nn as nn


class smallNet(nn.Module):
    def __init__(self, class_num):
        super(smallNet, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=1),  # 38*38
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),  # 20*20
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 10*10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),  # 10*10
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.feat = nn.Linear(3*10*10, 64)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num)
        )

        self.featout = False

    def set_featout(self, property):
        self.featout = property

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.shape[0], -1)
        feat = self.feat(x)
        x = self.classifier(feat)

        if self.featout:
            return x, feat
        else:
            return x
