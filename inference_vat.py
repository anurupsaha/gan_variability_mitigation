import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from models import Generator, Discriminator

BATCH_SIZE = 128
IMAGE_SIZE = 32
NC = 3          # Number of channels (RGB)
NZ = 100        # Size of z latent vector
NGF = 64        # Size of feature maps in generator
NDF = 64        # Size of feature maps in discriminator
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5     # Adam beta1
GAMMA = 0.99 
sigma_noise = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(NZ, NGF, NC).to(device)
netG.load_state_dict(torch.load('cifar10_generator_vat.pth'))
netG.eval()

fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
vutils.save_image(fake, f'reg_wo_perturb_vat.png', normalize=True)


with torch.no_grad():
    for name, param in netG.named_parameters():
        if len(param.data.shape) == 4:
            print(name, param.data.shape)
            cur_noise = torch.randn_like(param.data)
            param.data = param.data * (1 + sigma_noise * cur_noise)


with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
vutils.save_image(fake, f'reg_wi_perturb_vat.png', normalize=True)