"""Encoder-Decoder model for machine translation"""
"""May apply transformer."""

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


## LSTM version
## A vanilla LSTM encoder without the transformer architecture
class LSTM_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5, num_layers=2):
        super().__init__()
        # according to our Word2Vec model, out input size is 100
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)


    def forward(self, data):
        out, hidden = self.lstm(data.view(data.shape[0], data.shape[1], self.input_size))
        return out, hidden


class LSTM_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)


    def forward(self, data, hidden):
        out, hidden = self.lstm(data, hidden)
        out = self.linear(out)
        return out


# Combination of encoder and decoder
class LSTM_seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = LSTM_encoder(input_size, hidden_size, 0, 1)
        self.decoder = LSTM_decoder(input_size, hidden_size, 0, 1)


    def forward(self, data):
        data = data.clone().detach().requires_grad_(True).to(device)  
        code, hidden = self.encoder(data)
        output = self.decoder(code, hidden)
        return output



############################################################################

## GRU version

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, seq_input, hidden):
        embedded = self.embedding(seq_input).view(1, 1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def sample(self,seq_list):
        word_inds = torch.LongTensor(seq_list).to(device)
        h = self.initHidden()
        for word_tensor in word_inds:
            h = self(word_tensor,h)
        return h

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 10

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_input, hidden):
        output = self.embedding(seq_input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def sample(self,pre_hidden):
        inputs = torch.tensor([SOS_token], device=device)
        hidden = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            output,hidden = self(inputs,hidden)
            topv, topi = output.topk(1)
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            inputs = topi.squeeze().detach()
        return res


class GRU_seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size)

    def forward(self, source):
        hidden = self.encoder.initHidden()
        encoder_output, hidden = self.encoder(source, hidden)
        decoder_output, _ = self.decoder(encoder_output, hidden)
        return decoder_output




# learning_rate = 0.001
# hidden_size = 256

# encoder = EncoderRNN(len(lan1),hidden_size).to(device)
# decoder = DecoderRNN(hidden_size,len(lan2)).to(device)
# params = list(encoder.parameters()) + list(decoder.parameters())
# optimizer = optim.Adam(params, lr=learning_rate)

# loss = 0
# criterion = nn.NLLLoss()

# turns = 200

# print_every = 20
# print_loss_total = 0
# training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

# for turn in range(turns):
#     optimizer.zero_grad()
#     loss = 0
    
#     x,y = training_pairs[turn]
#     input_length = x.size(0)
#     target_length = y.size(0)

#     h = encoder.initHidden()
#     for i in range(input_length):
#         h = encoder(x[i],h)

#     decoder_input = torch.LongTensor([SOS_token]).to(device)
    
#     for i in range(target_length):
#         decoder_output,h = decoder(decoder_input,h)
#         topv, topi = decoder_output.topk(1)
#         decoder_input = topi.squeeze().detach()
#         loss += criterion(decoder_output, y[i])
#         if decoder_input.item() == EOS_token:break
                
#     print_loss_total += loss.item()/target_length
#     if (turn+1) % print_every == 0 :
#         print("loss:{loss:,.4f}".format(loss=print_loss_total/print_every))
#         print_loss_total = 0
        
#     loss.backward()
#     optimizer.step()

# def translate(s):
#     t = [lan1(i) for i in s.split()]
#     t.append(EOS_token)
#     f = encoder.sample(t)
#     s = decoder.sample(f)
#     r = [lan2.idx2word[i] for i in s]
#     return ' '.join(r)

# for pr in data:
#     print('>>',pr[0])
#     print('==',pr[1])
#     print('result:',translate(pr[0]))
#     print()
