#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import pickle as pkl


# In[2]:


from configuration import Config


# In[3]:


C = Config()


# In[8]:


def train_dataset():
    with open('data/train_acc.pkl', 'rb') as f:
        train_acc = pkl.load(f)     
    with open('data/train_ori.pkl', 'rb') as f:
        train_ori = pkl.load(f)     
    with open('data/train_smpl.pkl', 'rb') as f:
        train_smpl = pkl.load(f)     
        
    train_acc = np.array(train_acc)
    train_ori = np.array(train_ori)
    train_smpl = np.array(train_smpl)
    train_input = np.concatenate((train_ori, train_acc), axis=2)
    
    return train_acc, train_smpl, train_input


# In[9]:


def valid_dataset():
    with open('data/valid_acc.pkl', 'rb') as f:
        valid_acc = pkl.load(f)     
    with open('data/valid_ori.pkl', 'rb') as f:
        valid_ori = pkl.load(f)   
    with open('data/valid_smpl.pkl', 'rb') as f:
        valid_smpl = pkl.load(f)     
 
    return valid_acc, valid_ori, valid_smpl


# In[10]:


def test_dataset():
    with open('data/test_acc.pkl', 'rb') as f:
        test_acc = pkl.load(f)     
    with open('data/test_ori.pkl', 'rb') as f:
        test_ori = pkl.load(f)   
    with open('data/test_smpl.pkl', 'rb') as f:
        test_smpl = pkl.load(f)     

    return test_acc, test_ori, test_smpl


# In[ ]:




