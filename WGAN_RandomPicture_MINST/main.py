import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        x = torch.relu(self.map2(x))
        return torch.tanh(self.map3(x))

# 定义超参数
batch_size=100
input_size = 784
hidden_size = 256
output_size = 1

sample_dir = 'pic_data'

# 实例化生成器和判别器
G = Generator(input_size, hidden_size, output_size)

G.to(device)

G.load_state_dict(torch.load("generator.pt"))

G.eval()

noise = Variable(torch.randn(batch_size, input_size))
fake_images = G(noise)
'''fake_images = fake_images.view(-1, 28, 28)

# 保存生成的图像
torchvision.utils.save_image(fake_images.data, 'fake_images.png', nrow=10)'''

fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
torchvision.utils.save_image(denorm(fake_images), os.path.join(sample_dir, 'generated_images.png'))
print("done")

