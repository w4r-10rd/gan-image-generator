import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from models import Generator, Discriminator
from utils import get_data_loader

# Create the images directory if it does not exist
images_dir = 'images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
    print(f'Directory {images_dir} created.')
else:
    print(f'Directory {images_dir} already exists.')

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5
epochs = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataloader = get_data_loader(batch_size)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.size(0)  # Get the current batch size
        imgs = imgs.to(device)  # Move images to device
        
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        # Real images
        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Fake images
        z = torch.randn(batch_size, 100).to(device)  # Move z to device
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Combine losses and update Discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')

    # Save generated images
    image_path = os.path.join(images_dir, f'{epoch+1}.png')
    vutils.save_image(fake_imgs.data[:25], image_path, nrow=5, normalize=True)
    print(f'Images saved to {image_path}')
