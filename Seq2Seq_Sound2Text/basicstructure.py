# basicstructure.py
'''import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # Pass the input through the encoder
        #encoder_output, hidden = self.encoder(src)
        hidden = self.encoder(src)
        print(f"trg_shape:{trg.shape}")
        # Pass the encoder's final hidden state to the decoder
        output, _ = self.decoder(trg, hidden)
        return output
    
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads=8, dropout=0):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)


    def forward(self, src):
        src = self.dropout(src)
        _, hidden = self.gru(src)
        hidden = self.attention(hidden, hidden, hidden)[0]
        return hidden
    
class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, num_heads=8, dropout=0):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(output_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, trg, hidden):
        trg = self.dropout(trg)
        print(f"hidden_shape:{hidden.shape}")
        hidden1 = hidden
        idx = torch.tensor([0])
        hidden = hidden1.index_select(1, idx)
        hidden = hidden.squeeze(1)
        print(f"now hidden_shape:{hidden.shape}")
        #hidden = hidden.permute(1, 0)
        #print(f"Now hidden_shape:{hidden.shape}")
        output, hidden = self.gru(trg, hidden)
        output, _ = self.attention(output, output, output)
        output = self.fc(output)
        return output, hidden'''
