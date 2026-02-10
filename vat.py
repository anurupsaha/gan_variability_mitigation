import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from models import Generator, Discriminator

# --- Configuration ---
BATCH_SIZE = 128
IMAGE_SIZE = 32
NC = 3          # Number of channels (RGB)
NZ = 100        # Size of z latent vector
NGF = 64        # Size of feature maps in generator
NDF = 64        # Size of feature maps in discriminator
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5     # Adam beta1
GAMMA = 0.99    # Exponential LR decay factor
sigma_noise = 0.3

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 1. Data Loading & Augmentation ---
# Standard augmentation for GANs typically involves normalization and flipping.
# Heavy augmentation (like cropping) can sometimes introduce artifacts in generation.
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Download and load CIFAR10
dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- 2. Weights Initialization ---
# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Initialize Models
netG = Generator(NZ, NGF, NC).to(device)
netD = Discriminator(NDF, NC).to(device)

# Apply weights
netG.apply(weights_init)
netD.apply(weights_init)

# --- 4. Optimization ---

criterion = nn.BCELoss()

# Fixed noise for visualization stability during training
fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

# Establish convention for real and fake labels
real_label = 1.
fake_label = 0.

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

# Schedulers: Exponential LR Decay
schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=GAMMA)
schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=GAMMA)

# --- 5. Training Loop ---

print("Starting Training Loop...")

for epoch in range(NUM_EPOCHS):
    with torch.no_grad():
        for name, param in netG.named_parameters():
            if len(param.data.shape) == 4:
                print(name, param.data.shape)
                cur_noise = torch.randn_like(param.data)
                param.data = param.data * (1 + sigma_noise * cur_noise)
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, NZ, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # Combine losses and update D
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()

        # Print stats every 100 batches
        if i % 100 == 0:
            print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    # Step the schedulers at the end of each epoch
    schedulerD.step()
    schedulerG.step()
    
    
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
vutils.save_image(fake, f'fake_samples_epoch_{epoch}.png', normalize=True)

print("Training finished.")

# Save only the Generator's state dictionary
torch.save(netG.state_dict(), 'cifar10_generator_vat.pth')

print("Generator weights saved to 'cifar10_generator.pth'")