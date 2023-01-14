import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        x = torch.relu(self.map2(x))
        return torch.sigmoid(self.map3(x))

# 定义超参数
batch_size = 100
input_size = 784
hidden_size = 256
output_size = 1
num_epochs = 20
learning_rate = 0.0002

# 下载并加载 MNIST 数据集
mnist_data = dset.MNIST(root='.', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# 实例化生成器和判别器
G = Generator(input_size, hidden_size, output_size)
D = Discriminator(input_size, hidden_size, output_size)

try:
    if torch.cuda.is_available():
        G.load_state_dict(torch.load("generator_gpu.pt"))
        D.load_state_dict(torch.load("discriminator_gpu.pt"))
    else:
        G.load_state_dict(torch.load("generator.pt"))
        D.load_state_dict(torch.load("discriminator.pt"))
except Exception as e:
    print("No pretrained models are found")

# 定义损失函数和优化器
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

G.to(device)
D.to(device)

total_steps = len(mnist_loader)

print("Train start")
G.train()
D.train()
# 训练循环
for epoch in range(num_epochs):
    print(f"Epoch:{epoch+1}")
    for i, (real_images, _) in tqdm(enumerate(mnist_loader)):
        # 将输入转换为 PyTorch 变量
        real_images = Variable(real_images.view(-1, 28*28).to(device))
        real_labels = Variable(torch.ones(batch_size))
        fake_labels = Variable(torch.zeros(batch_size))

        # 训练判别器
        D.zero_grad()
        noise = Variable(torch.randn(batch_size, input_size).to(device))
        noise.to(device)
        fake_images = G(noise)
        real_outputs = D(real_images)
        fake_outputs = D(fake_images)
        D_loss = -torch.mean(torch.log(real_outputs) + torch.log(1 - fake_outputs))
        D_loss.backward()
        D_optimizer.step()

        # 训练生成器
        G.zero_grad()
        noise = Variable(torch.randn(batch_size, input_size))
        fake_images = G(noise.to(device))
        fake_outputs = D(fake_images.to(device))
        G_loss = -torch.mean(torch.log(fake_outputs))
        G_loss.backward()
        G_optimizer.step()
        
        if ((i+1)%100) == 0:
            tqdm.write(f"Epoch:{epoch+1}/{num_epochs}, Step:{i+1}/{total_steps}, G_loss:{G_loss}, D_loss:{D_loss}")
    print(f"Epoch:{epoch+1}/{num_epochs}, G_loss:{G_loss}, D_loss:{D_loss}")
    if ((epoch+1)%5) == 0:
        torch.save(G.state_dict(), 'generator.pt')
        torch.save(D.state_dict(), 'discriminator.pt')

# 保存训练好的模型
torch.save(G.state_dict(), 'generator.pt')
torch.save(D.state_dict(), 'discriminator.pt')
