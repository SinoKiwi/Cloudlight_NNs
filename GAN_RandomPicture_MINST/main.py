import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

# Hyper-parameters
latent_size = 100
hidden_size = 256
image_size = 784*3
num_epochs = 50
batch_size = 100
sample_dir = 'samples'

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G.to(device)

batch_size = 100
latent_size = 100
sample_dir = 'samples'

#使用模型
#加载模型参数
print("load_state_dict")
G.load_state_dict(torch.load('G.pt'))

#生成图像
print("generating")
z = torch.randn(batch_size, latent_size).to(device)
fake_images = G(z)

#保存图像
print("saving")
fake_images = fake_images.reshape(fake_images.size(0), 3, 28, 28)
torchvision.utils.save_image(denorm(fake_images), os.path.join(sample_dir, 'generated_images.png'))