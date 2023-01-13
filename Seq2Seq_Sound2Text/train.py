'''train.py
   训练
   先调用dataproc加载数据，创建Dataloader
   然后定义超参数，初始化模型
   使用遗传算法优化超参数TODO
   再进入训练循环，每5epoches设置一次保存模型
   最后保存模型并随机测试
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataproc import all_steps
from basicstructure import Encoder, Decoder
from basicstructure import Seq2Seq

#定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    src, trg = zip(*batch)
    src = torch.stack(src, dim=0)
    src = src.squeeze(2)
    src = src.permute(2,0,1)
    trg = torch.stack(trg, dim=0)
    #src = src.reshape(-1,batch_size,16000)
    return src, trg

#数据加载
batch_size=1
dataset, vocab_dict = all_steps('./data/train/audios/', "./data/train/train_debug.tsv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
#TODO: Repair Bug: Let it load changing-length sequence -> Possible Methoud:pack_padded_sequence&pad_packed_sequence;collate_fn

#超参数定义
input_dim = 128
encoder_hidden_dim = 300
encoder_layers = 3
decoder_input_dim = encoder_hidden_dim
decoder_layers = 3
decoder_output_dim = len(vocab_dict)
num_heads = 5
decoder_num_heads = 3
dropout = 0.3
learning_rate = 0.001

num_epochs = 10

#定义模型
encoder = Encoder(input_dim, encoder_hidden_dim, num_heads, encoder_layers, dropout)
decoder = Decoder(decoder_input_dim, decoder_output_dim, decoder_num_heads, decoder_layers, dropout)
model = Seq2Seq(encoder, decoder)

model.to(device)

# 定义损失函数
criterion = nn.CTCLoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

i=0

#TODO:padding for ctc ->maybe
print("Train start")
model.train()
for epochs in range(num_epochs):
   print("Epoch:", epochs+1)
   i=0
   for _, lst in tqdm(enumerate(dataloader)):
      #print("src:", src)
      #print("trg:", trg)
      src = lst[0]
      trg = lst[1]
      print("src_shape", src.shape)
      #print("trg_shape", trg.shape)
      #src = src.squeeze()
      #print("now src_shape:", src.shape)
      '''optimizer.zero_grad()
      output = model(src, trg)
      loss = criterion(output, trg)
      loss.backward()
      optimizer.step()'''
      encoder_hidden = torch.zeros(encoder_layers, src.size(0), encoder_hidden_dim)
      decoder_hidden = torch.zeros(decoder_layers, src.size(0), decoder_output_dim)
      optimizer.zero_grad()
      output, encoder_hidden, decoder_hidden = model(src, trg, encoder_hidden, decoder_hidden)
      loss = criterion(output.view(-1, input_dim), trg.view(-1))
      loss.backward()
      optimizer.step()
      i = i + 1
      print("A batch end.")
      print("epoch:{}/{}  batch:{}/{}  loss:{:.4f}".format(epochs,num_epochs,i,len(dataloader),loss))
   print("epoch:{}/{}  loss:{:.4f}".format(epochs,num_epochs,loss))
   if (epochs%5) == 0:
      torch.save(model.state_dict(), "model.pt")

torch.save(model.state_dict(), "model.pt")

