import os
import argparse


def main(opt):
    choice = ['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet']

    if opt.lr:
        for net in choice:
            os.system('python analysis.py --net {} --lr'.format(net))

    if opt.optim:
        for net in choice:
            os.system('python analysis.py --net {} --optim'.format(net))

    if opt.loss:
        for net in choice:
            os.system('python analysis.py --net {} --loss'.format(net))

    if opt.label0:
        for net in choice:
            os.system('python analysis.py --net {} --label0'.format(net))

    if opt.label1:
        for net in choice:
            os.system('python analysis.py --net {} --label1'.format(net))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action='store_true', default=False)
    parser.add_argument('--optim', action='store_true', default=False)
    parser.add_argument('--loss', action='store_true', default=False)
    parser.add_argument('--label0', action='store_true', default=False)
    parser.add_argument('--label1', action='store_true', default=False)
    opt = parser.parse_args()

    main(opt)
