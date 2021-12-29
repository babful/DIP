#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


class Config:
    def __init__(self):
        self.dict = self.make_dict()
        
    def make_dict(self):
        dictionary = {}
        
        dictionary['train_sample_num'] = 851
        dictionary['valid_sample_num'] = 3
        dictionary['test_sample_num'] = 18
        
        dictionary['batch_size'] = 16
        dictionary['num_workers'] = 1
        dictionary['lr'] = 0.001
        dictionary['weight_decay'] = 0.04
        
        dictionary['epochs'] = 100000
        dictionary['clip_grad'] = 1.0
        dictionary['print_every'] = 15
        
        dictionary['input_size'] = 60
        dictionary['hidden_size'] = 512
        dictionary['output_size'] = 150
        dictionary['n_layers'] = 2
        
        dictionary['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return dictionary
    
    def get(self, key):
        return self.dict[key]


# In[ ]:




