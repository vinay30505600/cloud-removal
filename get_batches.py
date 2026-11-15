#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import torch
import matplotlib.pyplot as plt


# In[61]:


def get_batches(step):
    step=step%1600
    data=np.load(f'image_{(step//32)+1}.npy')
    optical=torch.empty((4,2,3,256,256))
    sar=torch.empty((4,2,2,256,256))
    mask=torch.empty((4,2,1,256,256))
    for i in range(4):
        for j in range(2):
            optical[i,j,:,:,:]=torch.from_numpy(data[((step%32)*8)+(2*i)+j,0:3,:,:])
            sar[i,j,:,:,:]=torch.from_numpy(data[((step%32)*8)+(2*i)+j,3:5,:,:])
            mask[i,j,:,:,:]=torch.from_numpy(data[((step%32)*8)+(2*i)+j,5:6,:,:])
    input=optical*(1-mask)+mask        
    return input,sar,optical,mask       


