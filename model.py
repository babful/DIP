#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np


# In[2]:


class DIP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, drop_prob=0.5):
        super(DIP, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.input_layer = nn.Sequential(nn.Dropout(p=0.2),
                                         nn.Linear(input_size, hidden_size),
                                         nn.ReLU())
        
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=drop_prob,
                            bidirectional=True)
        
        self.output_layer_mu = nn.Sequential(nn.Linear(2 * hidden_size, output_size),
                                             nn.LogSoftmax(dim=2))
        self.output_layer_sigma = nn.Sequential(nn.Linear(2 * hidden_size, output_size),
                                                nn.Softplus())
        
    def forward(self, x, h):
        x = self.input_layer(x)
        lstm_out, h = self.lstm(x, h)
        output_mu = self.output_layer_mu(lstm_out)
        output_sigma = self.output_layer_sigma(lstm_out)
        
        smpl_mu, acc_mu = output_mu[:, :, :135], output_mu[:, :, 135:]
        smpl_sigma, acc_sigma = output_sigma[:, :, :135], output_sigma[:, :, 135:]
        
        return smpl_mu, acc_mu, smpl_sigma, acc_sigma
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(2 * self.n_layers, 
                             batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(2 * self.n_layers, 
                             batch_size, self.hidden_size).zero_().cuda())
        
        return hidden


# In[ ]:




