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

    test_acc_his = []
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for file_name in os.listdir('./visualize/hyper_param/data'):
        with open('./visualize/hyper_param/data/'+file_name, 'r') as fp:
            data = json.load(fp)
        train_loss = data['train_loss']
        val_acc = data['val_acc']
        test_acc = data['test_acc']
        test_acc_his.append(test_acc)

        epoch = np.arange(1, len(train_loss)+1)
        ax1.plot(epoch, train_loss)
        ax2.plot(epoch, val_acc)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('validation accuracy')
    ax1.legend(['lr='+str(x) for x in choice], loc='upper right')
    ax2.legend(['lr={}, test_acc={:.1f}%'.format(x, 100*y) for (x, y) in zip(choice, test_acc_his)], loc='lower right')
    fig.suptitle('Training performance based on different lr\nmodel: {}'.format(opt.net))
    plt.savefig('./visualize/hyper_param/lr/{}.jpg'.format(opt.net))


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

    test_acc_his = []
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for file_name in os.listdir('./visualize/hyper_param/data'):
        with open('./visualize/hyper_param/data/'+file_name, 'r') as fp:
            data = json.load(fp)
        train_loss = data['train_loss']
        val_acc = data['val_acc']
        test_acc = data['test_acc']
        test_acc_his.append(test_acc)

        epoch = np.arange(1, len(train_loss)+1)
        ax1.plot(epoch, train_loss)
        ax2.plot(epoch, val_acc)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('validation accuracy')
    ax1.legend(['optim='+str(x) for x in choice], loc='upper right')
    ax2.legend(['optim={}, test_acc={:.1f}%'.format(x, 100*y) for (x, y) in zip(choice, test_acc_his)], loc='lower right')
    fig.suptitle('Training performance based on different optimizers\nmodel: {}'.format(opt.net))
    plt.savefig('./visualize/hyper_param/optim/{}.jpg'.format(opt.net))


def wrong_label(opt):
    choice = [0.0, 0.1, 0.2, 0.3]
    if not os.path.exists('./visualize/hyper_param/label'):
        os.mkdir('./visualize/hyper_param/label')

    for file in os.listdir('./visualize/hyper_param/data'):
        os.remove('./visualize/hyper_param/data/'+file)

    print('>>> Analyze how wrong labels affect net training!')
    for prop in choice:
        print('------wrong label proportion = {}------'.format(prop))
        os.system('python main.py --net {} --wrong_prop {} --eval'.format(opt.net, prop))

    test_acc_his = []
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for file_name in os.listdir('./visualize/hyper_param/data'):
        with open('./visualize/hyper_param/data/'+file_name, 'r') as fp:
            data = json.load(fp)
        train_loss = data['train_loss']
        val_acc = data['val_acc']
        test_acc = data['test_acc']
        test_acc_his.append(test_acc)

        epoch = np.arange(1, len(train_loss)+1)
        ax1.plot(epoch, train_loss)
        ax2.plot(epoch, val_acc)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('validation accuracy')
    ax1.legend(['p='+str(x) for x in choice], loc='upper right')
    ax2.legend(['p={:.0f}%, test_acc={:.1f}%'.format(100*x, 100*y) for (x, y) in zip(choice, test_acc_his)], loc='lower right')
    fig.suptitle('Training performance under different wrong label proportion\nmodel: {}'.format(opt.net))
    plt.savefig('./visualize/hyper_param/label/{}.jpg'.format(opt.net))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', choices=['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet'], default='baseline')
    parser.add_argument('--lr', action='store_true', default=False)
    parser.add_argument('--optim', action='store_true', default=False)
    parser.add_argument('--label', action='store_true', default=False)
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
    if opt.label:
        wrong_label(opt)
