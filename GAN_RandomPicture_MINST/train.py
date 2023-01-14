import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 100
hidden_size = 256
image_size = 784*3
num_epochs = 30
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
# MINST dataset
mnist = torchvision.datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=batch_size, 
                                         shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

#load pretrained model
'''if os.path.exists("G.pt"):
    G.load_state_dict(torch.load("G.pt"))
if os.path.exists("D.pt"):
    D.load_state_dict(torch.load("D.pt"))'''

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

g_train_steps = 1

g_loss_last = 0
same_count = 0

# Start training
print("Train start")
total_step = len(data_loader)
for epoch in range(num_epochs):
    print(f"Epoch:{epoch+1}")
    for i, (images, _) in tqdm(enumerate(data_loader)):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are necessary for BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
           
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
    
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        
        try:
            if (d_loss.item() >= 0.25) and (g_loss.item() <= 0.2):
                d_optimizer.step()
        except Exception as e:
            if d_loss.item() >= 0.25:
                d_optimizer.step()

    
        for step in range(g_train_steps):
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
    
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
    
            # Backprop and optimize
            reset_grad()
            g_loss.backward()
            g_optimizer.step()
            
            '''if g_loss.round(decimals=4) == g_loss_last:
                same_count = same_count+1
            if same_count >= 5:
                d_optimizer.step()
            g_loss_last = g_loss.round(decimals=4)'''
            
        if ((i+1)%100) == 0:
            tqdm.write('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))

    print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
          .format(epoch+1, num_epochs, d_loss.item(), g_loss.item(), 
                  real_score.mean().item(), fake_score.mean().item()))
    #if((epoch+1)%10) == 0:
    torch.save(G.state_dict(), 'G.pt')
    torch.save(D.state_dict(), 'D.pt')

# Save real images
'''if (epoch+1) == 1:
    images = images.reshape(images.size(0), 3, 28, 28)
    save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))'''

# Save sampled images
fake_images = fake_images.reshape(fake_images.size(0), 3, 28, 28)
save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

#Save the model checkpoints
torch.save(G.state_dict(), 'G.pt')
torch.save(D.state_dict(), 'D.pt')
print("Train end")
