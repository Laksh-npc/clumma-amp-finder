import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.utils.rnn as rnn_utils
import pandas as pd

class Attention(nn.Module):
    def __init__(self,
                 activation='relu',
                 input_shape=None,
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 out_channels=8,
                 bias=True):
        super(Attention, self).__init__()

        # self.activation = torch.tanh()
        
        self.W_regularizer = W_regularizer
        self.u_regularizer = u_regularizer
        self.b_regularizer = b_regularizer
        
        self.W_constraint = W_constraint
        self.u_constraint = u_constraint
        self.b_constraint = b_constraint
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        self.activation=torch.nn.ReLU()

        amount_features = out_channels
        attention_size  = out_channels
        self.W = torch.empty((amount_features, attention_size))
        torch.nn.init.xavier_uniform_(self.W,  gain=nn.init.calculate_gain('relu'))

        # self.W1 = torch.empty((amount_features, attention_size))
        # torch.nn.init.xavier_uniform_(self.W1,  gain=nn.init.calculate_gain('relu'))

        # self.W2 = torch.empty((amount_features, attention_size))
        # torch.nn.init.xavier_uniform_(self.W2,  gain=nn.init.calculate_gain('relu'))
        if self.bias:
            self.b = torch.zeros(attention_size)
            # self.b1 = torch.zeros(attention_size)
            # self.b2 = torch.zeros(attention_size)
        else:
            self.register_parameter('b', None)
            # self.register_parameter('b1', None)
            # self.register_parameter('b2', None)

        # self.context1 = torch.empty((attention_size,))
        # torch.nn.init.uniform_(self.context1)

        self.context = torch.empty((attention_size,))
        torch.nn.init.uniform_(self.context)

        # self.context2 = torch.empty((attention_size,))
        # torch.nn.init.uniform_(self.context1)

        

    def forward(self, x, mask=None):        
        # U = tanh(H*W + b) (eq. 8)        


        # ui_1 = x @ self.W1              # (b, t, a)
        # if self.bias is not None:
        #     ui_1 += self.b1
        # ui_1= self.activation(ui_1)           # (b, t, a)


        ui = x @ self.W             # (b, t, a)
        if self.bias is not None:
            ui += self.b
        ui = self.activation(ui)           # (b, t, a)

        # ui_2 = x @ self.W2              # (b, t, a)
        # if self.bias is not None:
        #     ui_2 += self.b2
        # ui_2= self.activation(ui_2)           # (b, t, a)


        # Z = U * us (eq. 9)
        us = self.context.unsqueeze(0)   # (1, a)
        ui_us = ui @ us.transpose(0, 1)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = ui_us.squeeze(-1)  # (b, t, 1) -> (b, t)

        # us_1 = self.context1.unsqueeze(0)   # (1, a)
        # ui_us_1 = ui_1 @ us_1.transpose(0, 1)              # (b, t, a) * (a, 1) = (b, t, 1)
        # ui_us_1 = ui_us_1.squeeze(-1)  # (b, t, 1) -> (b, t)

        # us_2 = self.context2.unsqueeze(0)   # (1, a)
        # ui_us_2 = ui_2 @ us_2.transpose(0, 1)              # (b, t, a) * (a, 1) = (b, t, 1)
        # ui_us_2 = ui_us_2.squeeze(-1)  # (b, t, 1) -> (b, t)
        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = alpha.unsqueeze(-1)     # (b, t, 1)
        

        # alpha_2 = self._masked_softmax(ui_us_2, mask) # (b, t)
        # alpha_2 = alpha_2.unsqueeze(-1)     # (b, t, 1)
        # df = pd.DataFrame(alpha.detach().numpy()[0])
        # df.to_csv('Attention_{}.csv'.format(1))


        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return torch.sum(x * alpha, dim=1), alpha

    def _masked_softmax(self, logits, mask):

        """PyTorch's default implementation of softmax allows masking through the use of
        `torch.where`. This method handles masking if `mask` is not `None`."""
        
        # softmax(x):
        #    b = max(x)
        #    s_i = exp(xi - b) / exp(xj - b)
        #    return s

        b, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - b

        exped = torch.exp(logits)

        # ignoring masked inputs
        if mask is not None:
            mask = mask.float()
            exped *= mask

        partition = torch.sum(exped, dim=-1, keepdim=True)

        # if all timesteps are masked, the partition will be zero. To avoid this
        # issue we use the following trick:
        partition = torch.max(partition, torch.tensor(torch.finfo(logits.dtype).eps))

        return exped / partition

class EncoderSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Input x has shape (batch_size, sequence_length, d_model)
        
        # Linear projections to query (q), key (k), and value (v)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Split into multiple heads
        q = q.view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(x.size(0), -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.d_head ** 0.5)
        attention_weights = self.softmax(scores)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, v)

        # Merge heads
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, self.num_heads * self.d_head)

        # Apply a linear transformation
        output = self.fc(attended_values)

        return output


class SNNModel(nn.Module):
    def __init__(self, inp, in_channels = 26 , no_of_head=1, out_chan=16):

        super(SNNModel, self).__init__()

        self.Linear = nn.Linear(35, in_channels)

        # self.lstm  = torch.nn.LSTM(32, 16, 5, batch_first=True, bidirectional=True)

        # self.encoder = EncoderSelfAttention(in_channels, 8)

        self.conv1 = nn.Conv1d(in_channels, out_chan, 3, stride=1)
        self.conv2 = nn.Conv1d(in_channels, out_chan, 5, stride=1)
        self.conv3 = nn.Conv1d(in_channels, out_chan, 7, stride=1)
        self.attention_heads=[Attention(input_shape=inp, out_channels=out_chan) for i in range(no_of_head)]
        self.Dense1=nn.Linear(no_of_head*out_chan, 512)
        self.Dense2=nn.Linear(512, 512)
        self.Dense3=nn.Linear(512, 256)
        self.Dense4=nn.Linear(512, 512)
        self.Dense5=nn.Linear(1024, 1024)
        self.Output=nn.Linear(512, 1)
        self.sigm=torch.nn.Sigmoid()
        self.activation=nn.LeakyReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.bacthnorm=nn.BatchNorm1d(512)
        self.bacthnorm1=nn.BatchNorm1d(256)


    def forward(self, x):

        
        x = self.activation(self.Linear(x))

        # x = self.encoder(x)
        
        x = x.transpose(1, 2)       

        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)


        x = torch.cat((x1, x2, x3), dim=2)
        x = x.transpose(1, 2)

        

        # x = self.activation(self.lstm(x)[0])
        
        x_l=[]
        for ind, fun in enumerate(self.attention_heads):
            x_pred, scores = fun(x) 
            x_l.append(x_pred)
            scores = torch.squeeze(x_pred)
            df = pd.DataFrame(scores.detach().numpy())
            df.to_csv('Attention_{}.csv'.format(ind))

        x=torch.cat(tuple(x_l), dim=1)

        x=self.bacthnorm(self.Dense1(x))
        x=self.activation(x)
        # x=self.dropout(x)
        # x=self.bacthnorm(self.Dense2(x))
        # x=self.activation(x)
        # x=self.bacthnorm1(self.Dense3(x))
        # x=self.dropout(x)
        # x=self.activation(x)
        # x=self.bacthnorm(self.Dense4(x))
        # x=self.activation(x)
        # x=self.dropout(x)
        # x=self.bacthnorm(self.Dense5(x))
        # x=self.activation(x)
        x=self.dropout(x)
        x=self.Output(x)
        out = self.sigm(x)

        return out

	