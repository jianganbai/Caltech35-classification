import torch.nn as nn
import torchvision.models as models


def densenet(class_num):
    net = models.densenet121(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.classifier = nn.Linear(1024, class_num)
    # print(net)
    return net


if __name__ == '__main__':
    densenet(35)
