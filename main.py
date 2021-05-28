import numpy as np
from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.transforms.transforms import RandomVerticalFlip

import argparse
import time
import copy
import json
import os

import visualize
from model.base_model import base_model
from model.base_model_dropout import base_model_dropout
from model.small_CNN import smallNet
from model.resnet import resnet
from model.densenet import densenet


def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config['imsize'], interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config['imsize'], interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if config['add_real']:
        train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])
    else:
        train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'], wrong_prop=config['wrong_prop'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['net'] == 'baseline':
        model = base_model(class_num=config['class_num'])
    elif config['net'] == 'baseline-dropout':
        model = base_model_dropout(class_num=config['class_num'])
    elif config['net'] == 'small-CNN':
        model = smallNet(class_num=config['class_num'])
    elif config['net'] == 'resnet':
        model = resnet(class_num=config['class_num'])
    elif config['net'] == 'densenet':
        model = densenet(class_num=config['class_num'])
    model.to(device)

    if config['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
        if config['L2']:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1, last_epoch=-1)

    if config['MSE']:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_loss, train_acc, val_acc = train(config, train_loader, val_loader, model, optimizer, scheduler, criterion, device)

    model.load_state_dict(torch.load('./model/{}.pth'.format(config['net'])))
    test_acc = test(test_loader, model, device, visual=True, config=config, data_type='test')
    if config['train_tsne']:
        _ = test(train_loader, model, device, visual=True, config=config, data_type='train')
    print('===========================')
    print("test accuracy:{}%".format(test_acc * 100))
    return train_loss, train_acc, val_acc, test_acc


def config_print(device, config):
    if config['L1']:
        loss_func = 'CrossEntropy+L1'
    elif config['L2']:
        loss_func = 'CrossEntropy+L2'
    elif config['MSE']:
        loss_func = 'MSE'
    else:
        loss_func = 'CrossEntropy'
    addition = ''
    if config['add_real']:
        addition += 'add_real; '
    if config['wrong_prop'] > 0:
        addition += 'wrong_prop={}; '.format(config['wrong_prop'])

    message = '[Device: {}] [Epoch: {}] [Net Type: {}] [Loss: {}] [Addition: {}]'.format(
        device, config['epochs'], config['net'], loss_func, addition)
    print(message)
    print('===========================')


def train(config, train_loader, val_loader, model, optimizer, scheduler,
          criterion, device, class_num=35):
    train_loss_his, train_acc_his, val_acc_his = [], [], []
    best_val_acc = 0
    ones = np.eye(class_num)
    config_print(device, config)

    for epoch in np.arange(1, config['epochs']+1):
        model.train()
        epoch_start = time.time()
        train_acc, num, epoch_loss = 0, 0, 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            if config['L1']:
                param_loss = 0
                for param in model.parameters():
                    param_loss += torch.sum(torch.abs(param))
                loss = criterion(output, label)+(1e-3)*param_loss
            elif config['MSE']:
                one_hot = ones[label.cpu().numpy(), :]
                one_hot = torch.FloatTensor(one_hot)
                loss = criterion(output, one_hot.to(device))
            else:
                loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num += data.shape[0]
            train_acc += (label == output.argmax(dim=1)).sum().cpu().item()
            epoch_loss += loss.cpu().item()

        epoch_loss = epoch_loss/(batch_idx+1)
        train_acc = train_acc/num
        val_acc = test(val_loader, model, device)
        train_loss_his.append(epoch_loss)
        train_acc_his.append(train_acc)
        val_acc_his.append(val_acc)
        time_len = time.time()-epoch_start
        print('[Epoch: {}/{}] [Loss: {:.6f}] [Train Acc: {:.6f}] [Val Acc: {:.6f}] [Time: {:.1f}sec]'.format(
            epoch, config['epochs'], epoch_loss, train_acc, val_acc, time_len))
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_net_para = copy.deepcopy(model.state_dict())

    print('Training Complete! Best validation accuracy: {:.6f}'.format(best_val_acc))
    torch.save(best_net_para, './model/{}.pth'.format(config['net']))
    return train_loss_his, train_acc_his, val_acc_his


def test(data_loader, model, device, visual=False, config=None, data_type=None):
    model.eval()
    correct = 0
    if visual:
        feat_all, label_all = np.zeros((0, 64)), np.zeros((0))
        model.set_featout(True)
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            if visual:
                output, feat = model(data)
            else:
                output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().cpu()

            if visual:
                feat_all = np.row_stack((feat_all, feat.cpu().numpy()))
                label_all = np.concatenate([label_all, label.cpu().numpy()])
    accuracy = correct.item() * 1.0 / len(data_loader.dataset)

    # t-sne可视化
    if visual:
        visualize.tsne_vis(feat_all, label_all, config, data_type)
        model.set_featout(False)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])
    parser.add_argument('--net', choices=['baseline', 'baseline-dropout', 'small-CNN', 'resnet', 'densenet'], default='small-CNN')
    parser.add_argument('--optim', choices=['SGD', 'RMSprop', 'Adam'], default='SGD')
    parser.add_argument('--L1', action='store_true', default=False)
    parser.add_argument('--L2', action='store_true', default=False)
    parser.add_argument('--MSE', action='store_true', default=False)
    parser.add_argument('--add_real', action='store_true', default=False)
    parser.add_argument('--wrong_prop', type=float, default=0.0)
    parser.add_argument('--train_tsne', action='store_true', default=True)
    parser.add_argument('--eval', action='store_true', default=False)
    config = parser.parse_args()

    config = vars(config)
    if config['net'] == 'resnet' or config['net'] == 'densenet':
        config['imsize'] = [224, 224]
    elif config['net'] == 'baseline' or 'baseline-dropout':
        config['imsize'] = [112, 112]

    train_loss, train_acc, val_acc, test_acc = main(config)
    if not config['eval']:
        visualize.loss_and_acc(train_loss, train_acc, val_acc, config)
    else:
        data = {'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc}
        prefix = list(map(lambda x: int(x[0]), list(os.listdir('./visualize/hyper_param/data'))))
        if len(prefix) == 0:
            file_name = '0.json'
        else:
            file_name = '{}.json'.format(max(prefix)+1)
        with open('./visualize/hyper_param/data/'+file_name, 'w') as fp:
            json.dump(data, fp)
