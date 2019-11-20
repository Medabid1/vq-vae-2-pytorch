import torch
import numpy as np 
import matplotlib.pyplot as plt 

from torch import nn, optim
from model import UnifModel
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, SubsetRandomSampler


model = UnifModel()
model.load_state_dict(torch.load('checkpoint/vqvae_012.pt'))


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
svhn = datasets.SVHN(root='./svhn', transform=s_transform, split='train')
mnist = datasets.MNIST(root='./mnist', transform=m_transform)
s = svhn.__getitem__(999)
#plt.imshow((img[0].permute(1,2,0).numpy()*255).astype(np.uint8))
#s = torch.zeros((1,3,32,32))
s = s[0].unsqueeze(0)
mloader = DataLoader(mnist, batch_size=10, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(range(10)))
s = torch.cat([s]*10, 0)
model = nn.DataParallel(model).to(device)
model.eval()
onlyTop = False
onlyBot = False
i = 1
for j, (m,_) in enumerate(mloader):    
    m = m.to(device)
    s = s.to(device)
    mout, sout, _ = model(m,s, onlyTop, onlyBot)

    utils.save_image(
                torch.cat([m, mout], 0),
                f'test/all_data_mnist_top:{str(onlyTop).zfill(5)}_bot :{str(onlyBot).zfill(5)}.png',
                nrow=10,
                normalize=True,
                range=(-1, 1),
            )
    utils.save_image(
        torch.cat([s, sout], 0),
        f'test/all_data_svhn_top:{str(onlyTop).zfill(5)}_bot :{str(onlyBot).zfill(5)}.png',
        nrow=10,
        normalize=True,
        range=(-1, 1),
    )
    if j > i : 
        break 

