from __future__ import print_function
import argparse
import random
import numpy as np
import time

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from dataset import tiny_caltech35
import dcgan

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--save_interval', type=int, default=50)
parser.add_argument('--lambda_gp', type=float, default=10)
opt = parser.parse_args()

if not os.path.exists('./generated'):
    os.mkdir('./generated')
if not os.path.exists('./archive'):
    os.mkdir('./archive')

# 设定随机初始化种子
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

transform_train = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])

dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)


def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = dcgan.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, 1, opt.n_extra_layers)
netG.apply(weights_init)  # 初始化网络参数

netD = dcgan.DCGAN_D(opt.imageSize, opt.nz, opt.nc, opt.ndf, 1, opt.n_extra_layers)
netD.apply(weights_init)

netD = netD.to(device)
netG = netG.to(device)

optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

Tensor = torch.cuda.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


batch_done = 0
print('======training start======')
for epoch in range(opt.epoch):
    epoch_start = time.time()
    loss_D_aver, loss_G_aver = 0, 0
    for batch_id, (imgs, _) in enumerate(dataloader):
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz, 1, 1))))
        imgs = imgs.to(device)
        # 先训练D
        for _ in range(opt.Diters):
            optimizerD.zero_grad()
            fake = netG(z)
            gradient_penalty = compute_gradient_penalty(netD, imgs.data, fake.data)
            loss_D = -torch.mean(netD(imgs))+torch.mean(netD(fake.detach()))+opt.lambda_gp*gradient_penalty
            loss_D.backward()
            optimizerD.step()

        # 再训练G
        optimizerG.zero_grad()
        fake = netG(z)
        loss_G = -torch.mean(netD(fake))
        loss_G.backward()
        optimizerG.step()

        loss_G_aver += loss_G.cpu().item()
        loss_D_aver += loss_D.cpu().item()
        if batch_done % opt.save_interval == 0:
            save_image(fake.data[:16], './generated/{}.jpg'.format(batch_done), nrow=4, normalize=True)
        batch_done += 1

    loss_D_aver /= batch_id+1
    loss_G_aver /= batch_id+1
    epoch_time = time.time()-epoch_start
    print('[epoch: {}/{}] [loss_D: {}] [loss_G: {}] [time: {:.1f}sec]'.format(
            epoch, opt.epoch, loss_D_aver, loss_G_aver, epoch_time
        ))
