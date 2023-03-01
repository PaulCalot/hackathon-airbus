import torch
import numpy as np

def get_l_out(l_in, kernel_size, padding=0, dilation=1, stride=1):
    return np.floor((l_in + 2 * padding - dilation * (kernel_size - 1) -1)/stride + 1)

class ConvBlock1d(torch.nn.Module):
    def __init__(self, conv_kwargs, pool_kwargs, dropout_rate) -> None:
        super().__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.conv = torch.nn.Conv1d(**conv_kwargs)
        self.activation_fn = torch.nn.ReLU()
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        self.pooling = torch.nn.MaxPool1d(**pool_kwargs)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation_fn(x)
        x = self.pooling(x)
        x = self.dropout(x)
        return x

class Cnn1d(torch.nn.Module):
    def __init__(self, block_kwargs_list, linear_kwargs) -> None: # use_dv_head=False, use_date_head=False
        super().__init__()
        self.block_kwargs_list = block_kwargs_list
        ll = []
        for block_kwargs in block_kwargs_list:
            ll.append(ConvBlock1d(**block_kwargs))

        self.convnet = torch.nn.Sequential(*ll)
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fcnn = torch.nn.Linear(**linear_kwargs)
        self.activation_fn = torch.nn.ReLU()

        self.classification_head = torch.nn.Sequential(*[
            torch.nn.Linear(linear_kwargs['out_features'], 1),
            # torch.nn.Softmax(dim=-1)
        ])
        
    def forward(self, x):
        # main model
        x = self.convnet(x)
        x = torch.flatten(x, start_dim=1) # size : batch size x length
        x = self.fcnn(x)
        embedding = self.activation_fn(x)

        # classification head
        c = self.classification_head(x)
        return torch.squeeze(c), embedding