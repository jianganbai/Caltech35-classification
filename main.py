import numpy as np
from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import copy

import visualize
from model.base_model import base_model
from model.base_model_modify import base_model_modify
from model.resnet import resnet
from model.densenet import densenet


def main(config, param):
    transform_train = transforms.Compose([
        transforms.Resize(param['imsize'], interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(param['imsize'], interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
    # if you want to add the addition set and validation set to train
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.net == 'baseline':
        model = base_model(class_num=config.class_num)
    elif config.net == 'baseline-modify':
        model = base_model_modify(class_num=config.class_num)
    elif config.net == 'resnet':
        model = resnet(class_num=config.class_num)
    elif config.net == 'densenet':
        model = densenet(class_num=config.class_num)
    model.to(device)

    if config.L2:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_loss, train_acc, val_acc = train(config, train_loader, val_loader, model, optimizer, scheduler, criterion, device)

    # you can use validation dataset to adjust hyper-parameters
    model.load_state_dict(torch.load('./model/{}.pth'.format(config.net)))
    test_accuracy = test(test_loader, model, device, visual=True, config=config)
    print('===========================')
    print("test accuracy:{}%".format(test_accuracy * 100))
    visualize.loss_and_acc(train_loss, train_acc, val_acc, config)


def train(config, train_loader, val_loader, model, optimizer, scheduler, criterion, device):
    train_loss_his, train_acc_his, val_acc_his = [], [], []
    best_val_acc = 0
    print('Training on {} for {} epochs. Net type: {}'.format(device, config.epochs, config.net))
    print('===========================')
    for epoch in np.arange(1, config.epochs+1):
        model.train()
        epoch_start = time.time()
        train_acc, num, epoch_loss = 0, 0, 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            if config.L1:
                param_loss = 0
                for param in model.parameters():
                    param_loss += torch.sum(torch.abs(param))
                loss = criterion(output, label)+(1e-5)*param_loss
            else:
                loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num += data.shape[0]
            train_acc += (label == output.argmax(dim=1)).sum().cpu()
            epoch_loss += loss.cpu().item()

        epoch_loss = epoch_loss/(batch_idx+1)
        train_acc = train_acc/num
        val_acc = test(val_loader, model, device)
        train_loss_his.append(epoch_loss)
        train_acc_his.append(train_acc)
        val_acc_his.append(val_acc)
        time_len = time.time()-epoch_start
        print('[Epoch: {}/{}] [Loss: {:.6f}] [Train Acc: {:.6f}] [Val Acc: {:.6f}] [Time: {:.1f}sec]'.format(epoch, config.epochs, epoch_loss, train_acc, val_acc, time_len))
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_net_para = copy.deepcopy(model.state_dict())

    print('Training Complete! Best validation accuracy: {:.6f}'.format(best_val_acc))
    torch.save(best_net_para, './model/{}.pth'.format(config.net))
    return train_loss_his, train_acc_his, val_acc_his


def test(data_loader, model, device, visual=False, feat_num=64, config=None):
    model.eval()
    correct = 0
    if visual:
        feat_all, label_all = np.zeros((0, feat_num)), np.zeros((0))
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
        visualize.tsne_vis(feat_all, label_all, config)
        model.set_featout(False)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])
    parser.add_argument('--net', choices=['baseline', 'baseline-modify', 'resnet', 'densenet'], default='densenet')
    parser.add_argument('--L1', action='store_true', default=False)
    parser.add_argument('--L2', action='store_true', default=False)
    config = parser.parse_args()

    param = {}
    if config.net == 'resnet' or config.net == 'densenet':
        param['imsize'] = [224, 224]
    elif config.net == 'baseline' or 'baseline-modify':
        param['imsize'] = [112, 112]
    main(config, param)
