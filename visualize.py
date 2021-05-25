import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def tsne_vis(feat_all, label_all, config, vis_num=5):
    if not os.path.exists('./visualize/'):
        os.mkdir('./visualize/')
    if not os.path.exists('./visualize/feat_distri/'):
        os.mkdir('./visualize/feat_distri/')

    vis_type = list(range(vis_num))
    embedded = TSNE(n_components=2).fit_transform(feat_all)

    plt.figure()
    for label in vis_type:
        idx = np.where(label_all == label)[0].tolist()
        em = embedded[idx, :]
        plt.scatter(em[:, 0], em[:, 1])
    plt.title('feature extracted by {}, visualized by t-sne'.format(config['net']))
    plt.savefig('./visualize/feat_distri/{}.jpg'.format(config['net']))


def loss_and_acc(train_loss, train_acc, val_acc, config):
    if not os.path.exists('./visualize/'):
        os.mkdir('./visualize/')
    if not os.path.exists('./visualize/loss&acc/'):
        os.mkdir('./visualize/loss&acc/')

    epoch = np.arange(1, config['epochs']+1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(epoch, train_loss, color='red', label='loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(epoch, train_acc, label='train_acc')
    ax2.plot(epoch, val_acc, label='val_acc')
    ax2.set_ylabel('accuracy')

    fig.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    plt.title('{} performance'.format(config['net']))
    plt.savefig('./visualize/loss&acc/{}.jpg'.format(config['net']))
