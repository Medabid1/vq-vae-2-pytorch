import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from  model import UnifModel
from scheduler import CycleScheduler


def train(epoch, mloader,sloader, model, optimizer, scheduler, device):
    #loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, imgs in enumerate(zip(mloader, sloader)):
        (mnist_img, _), (svhn_img, _) = imgs
        model.zero_grad()
        if mnist_img.size(0) != svhn_img.size(0) :
            m = min((mnist_img.size(0),svhn_img.size(0)))
            mnist_img = mnist_img[:m, :, :,:]
            svhn_img = svhn_img[:m, :, :,:]

        mnist_img = mnist_img.to(device)
        svhn_img = svhn_img.to(device)

        m_out, s_out, latent_loss = model(mnist_img, svhn_img)
        recon_loss = criterion(m_out, mnist_img) + criterion(s_out, svhn_img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * mnist_img.shape[0] * 2
        mse_n += mnist_img.shape[0] * 2

        lr = optimizer.param_groups[0]['lr']

        print((f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}', 
               f'latent: {latent_loss.item():.3f} - avg mse: {mse_sum / mse_n:.5f}', 
               f'lr: {lr:.5f}'))

        if i % 100 == 0:
            model.eval()

            msample = mnist_img[:sample_size]
            ssample = svhn_img[:sample_size]
            with torch.no_grad():
                mout, sout, _ = model(msample,ssample)

            utils.save_image(
                torch.cat([msample, mout], 0),
                f'sample/mnist{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                torch.cat([ssample, sout], 0),
                f'sample/svhn{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    m_transform = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    s_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    svhn = datasets.SVHN(root='./svhn', transform=s_transform, download=True, split='train')
    mnist = datasets.MNIST(root='./mnist', transform=m_transform, download=True)
    print(len(mnist), len(svhn))
    mloader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=4)
    sloader = DataLoader(svhn, batch_size=64, num_workers=4)#, sampler=SubsetRandomSampler(range(1000)))
    model = nn.DataParallel(UnifModel()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    #if args.sched == 'cycle':
    #   scheduler = CycleScheduler(
    #        optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
    #    )

    for i in range(args.epoch):
        train(i, mloader, sloader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
