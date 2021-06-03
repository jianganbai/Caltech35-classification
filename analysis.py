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
        if opt.add_real:
            os.system('python main.py --net {} --epochs {} --lr {} --eval --add_real'
                      .format(opt.net, opt.epochs, lr))
        else:
            os.system('python main.py --net {} --epochs {} --lr {} --eval'
                      .format(opt.net, opt.epochs, lr))

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
        if opt.add_real:
            os.system('python main.py --net {} --epochs {} --optim {} --eval --add_real'
                      .format(opt.net, opt.epochs, optim))
        else:
            os.system('python main.py --net {} --epochs {} --optim {} --eval'
                      .format(opt.net, opt.epochs, optim))

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
        if opt.add_real:
            os.system('python main.py --net {} --epochs {} --wrong_prop {} --eval --add_real'
                      .format(opt.net, opt.epochs, prop))
        else:
            os.system('python main.py --net {} --epochs {} --wrong_prop {} --eval'
                      .format(opt.net, opt.epochs, prop))

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


def loss_choice(opt):
    choice = ['CrossEntropy', 'L1', 'L2', 'MSE']
    if not os.path.exists('./visualize/loss_func'):
        os.mkdir('./visualize/loss_func')

    for file in os.listdir('./visualize/hyper_param/data'):
        os.remove('./visualize/hyper_param/data/'+file)

    print('>>> Analyze how choices of loss functions affect net training!')
    for loss in choice:
        print('------loss function = {}------'.format(loss))
        if loss == 'CrossEntropy':
            if opt.add_real:
                os.system('python main.py --net {} --epochs {} --eval --add_real'
                          .format(opt.net, opt.epochs))
            else:
                os.system('python main.py --net {} --epochs {} --eval'
                          .format(opt.net, opt.epochs))
        else:
            if opt.add_real:
                os.system('python main.py --net {} --epochs {} --{} --eval --add_real'
                          .format(opt.net, opt.epochs, loss))
            else:
                os.system('python main.py --net {} --epochs {} --{} --eval'
                          .format(opt.net, opt.epochs, loss))

    test_acc_his = []
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for file_name in os.listdir('./visualize/hyper_param/data/'):
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
    ax1.legend(['func='+str(x) for x in choice], loc='upper right')
    ax2.legend(['func={}, test_acc={:.1f}%'.format(x, 100*y) for (x, y) in zip(choice, test_acc_his)], loc='lower right')
    fig.suptitle('Training performance based on different loss functions\nmodel: {}'.format(opt.net))
    plt.savefig('./visualize/loss_func/{}.jpg'.format(opt.net))


def wrong_label_solution1(opt):  # clean dataset -> dirty dataset
    choice = [0.0, 0.1, 0.2, 0.3]
    if not os.path.exists('./visualize/hyper_param/label'):
        os.mkdir('./visualize/hyper_param/label')

    for file in os.listdir('./visualize/hyper_param/data'):
        os.remove('./visualize/hyper_param/data/'+file)

    print('>>> Analyze how wrong labels affect net training!')
    for prop in choice:
        print('------wrong label proportion = {}------'.format(prop))
        os.system('python cross_train.py --net {} --ep1 {} --ep2 {} --wrong_prop {} --eval'
                  .format(opt.net, opt.ep1, opt.ep2, prop))

    test_acc_his = []
    loss_max = 0
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

        if loss_max < max(train_loss):
            loss_max = max(train_loss)
        epoch = np.arange(1, len(train_loss)+1)
        ax1.plot(epoch, train_loss)
        ax2.plot(epoch, val_acc)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('validation accuracy')
    ax1.legend(['p='+str(x) for x in choice], loc='upper right')
    ax2.legend(['p={:.0f}%, test_acc={:.1f}%'.format(100*x, 100*y) for (x, y) in zip(choice, test_acc_his)], loc='lower right')

    separate_x = (opt.ep1+0.5) * np.ones(len(train_loss))
    separate_y = np.linspace(0, loss_max+0.1, num=len(train_loss))
    ax1.plot(separate_x, separate_y, linestyle=':')
    ax1.text(0.9*(opt.ep1+0.5), 0.9*loss_max, 'clean dataset', ha='right')
    ax1.text(1.1*(opt.ep1+0.5), 0.9*loss_max, 'dirty dataset', ha='left')
    ax2.plot(separate_x, np.linspace(0, 1, num=len(train_loss)), linestyle=':')
    ax2.text(0.9*(opt.ep1+0.5), max(test_acc_his)+0.2, 'clean dataset', ha='right')
    ax2.text(1.1*(opt.ep1+0.5), max(test_acc_his)+0.2, 'dirty dataset', ha='left')
    fig.suptitle('Training performance under different wrong label proportion\n\
    model: {} strategy: clean dataset -> dirty dataset'.format(opt.net))
    plt.savefig('./visualize/hyper_param/label/{}.jpg'.format(opt.net))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', choices=['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet'], default='baseline')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', action='store_true', default=False)
    parser.add_argument('--optim', action='store_true', default=False)
    parser.add_argument('--label0', action='store_true', default=False)
    parser.add_argument('--label1', action='store_true', default=False, help='counter measure 1')
    parser.add_argument('--loss', action='store_true', default=False)
    parser.add_argument('--add_real', action='store_true', default=False)
    parser.add_argument('--ep1', type=int, default=20)
    parser.add_argument('--ep2', type=int, default=60)
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
    if opt.label0:
        wrong_label(opt)
    if opt.loss:
        loss_choice(opt)
    if opt.label1:
        wrong_label_solution1(opt)
