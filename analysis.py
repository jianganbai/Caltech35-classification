import os
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np


def lr_choice(opt):
    choice = [0.1, 0.01, 0.001, 0.0001]
    if not os.path.exists('./visualize/hyper_param/lr'):
        os.mkdir('./visualize/hyper_param/lr')

    for file in os.listdir('./visualize/hyper_param/data'):
        os.remove('./visualize/hyper_param/data/'+file)

    print('>>> Analyze how lr affects net training!')
    for lr in choice:
        print('------lr = {}------'.format(lr))
        os.system('python main.py --net {} --lr {} --eval'.format(opt.net, lr))

    fig, ax = plt.subplots()
    for file_name in os.listdir('./visualize/hyper_param/data'):
        with open('./visualize/hyper_param/data/'+file_name, 'r') as fp:
            data = json.load(fp)
        train_loss = data['train_loss']
        epoch = np.arange(1, len(train_loss)+1)
        ax.plot(epoch, train_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train loss')
    ax.set_title('Training performance based on different lr\nmodel: {}'.format(opt.net))
    plt.legend(['lr='+str(x) for x in choice], loc='upper right')
    plt.savefig('./visualize/hyper_param/lr/loss-{}'.format(opt.net))


def optim_choice(opt):
    choice = ['SGD', 'RMSprop', 'Adam']
    if not os.path.exists('./visualize/hyper_param/optim'):
        os.mkdir('./visualize/hyper_param/optim')

    for file in os.listdir('./visualize/hyper_param/data'):
        os.remove('./visualize/hyper_param/data/'+file)

    print('>>> Analyze how choices of optimizer affect net training!')
    for optim in choice:
        print('------optim = {}------'.format(optim))
        os.system('python main.py --net {} --optim {} --eval'.format(opt.net, optim))

    fig, ax = plt.subplots()
    for file_name in os.listdir('./visualize/hyper_param/data'):
        with open('./visualize/hyper_param/data/'+file_name, 'r') as fp:
            data = json.load(fp)
        train_loss = data['train_loss']
        epoch = np.arange(1, len(train_loss)+1)
        ax.plot(epoch, train_loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train loss')
    ax.set_title('Training performance based on different optimizers\nmodel: {}'.format(opt.net))
    plt.legend(choice, loc='upper right')
    plt.savefig('./visualize/hyper_param/optim/loss-{}'.format(opt.net))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', choices=['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet'], default='baseline')
    parser.add_argument('--lr', action='store_true', default=False, help='how lr affects net training')
    parser.add_argument('--optim', action='store_true', default=True, help='how optimizer affects net training')
    opt = parser.parse_args()

    if not os.path.exists('./visualize'):
        os.mkdir('./visualize')
    if not os.path.exists('./visualize/hyper_param'):
        os.mkdir('./visualize/hyper_param')
    if not os.path.exists('./visualize/hyper_param/data'):
        os.mkdir('./visualize/hyper_param/data')

    if opt.lr:
        lr_choice(opt)
    if opt.optim:
        optim_choice(opt)
