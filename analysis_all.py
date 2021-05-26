import os

if __name__ == '__main__':
    choice = ['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet']

    # lr
    for net in choice:
        os.system('python analysis.py --net {} --lr'.format(net))
    # optim
    for net in choice:
        os.system('python analysis.py --net {} --optim'.format(net))
    # wrong label
    for net in choice:
        os.system('python analysis.py --net {} --label'.format(net))
