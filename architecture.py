#!/usr/bin/env python
# coding: utf-8

# In[73]:


import torch 
import torch.nn as nn
import torch.nn.functional as f


# In[74]:


class ConvBlock(nn.Module):
    def __init__(self,In_channels,Out_channels,Kernel_size,Stride,Padding,device,dtype):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=In_channels,out_channels=Out_channels,kernel_size=Kernel_size,stride=Stride,padding=Padding,device=device,dtype=dtype)
        self.instancenorm1=nn.InstanceNorm2d(num_features=Out_channels,eps=1e-5,affine=True,device=device,dtype=dtype)
        self.conv2=nn.Conv2d(in_channels=Out_channels,out_channels=Out_channels,kernel_size=Kernel_size,stride=Stride,padding=Padding,device=device,dtype=dtype)
        self.instancenorm2=nn.InstanceNorm2d(num_features=Out_channels,eps=1e-5,affine=True,device=device,dtype=dtype)
        self.activation=nn.LeakyReLU(negative_slope=0.2,inplace=False)
    def forward(self,x):
        return self.activation(self.instancenorm2(self.conv2(self.activation(self.instancenorm1((self.conv1(x)))))))





# In[75]:


class Encoder(nn.Module):
    def __init__(self,ConvBlock,In_channels,Out_channels1,Out_channels2,Out_channels3,Kernel_size,Stride,Padding,device,dtype):
        super().__init__()
        self.ConvBlock1=ConvBlock(In_channels,Out_channels1,Kernel_size,Stride,Padding,device,dtype)
        self.ConvBlock2=ConvBlock(Out_channels1,Out_channels2,Kernel_size,Stride,Padding,device,dtype)
        self.ConvBlock3=ConvBlock(Out_channels2,Out_channels3,Kernel_size,Stride,Padding,device,dtype)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        skip_1=self.ConvBlock1(x)
        out=self.pool(skip_1)
        skip_2=self.ConvBlock2(out)
        out=self.pool(skip_2)
        skip_3=self.ConvBlock3(out)
        out=self.pool(skip_3)
        return skip_1,skip_2,skip_3,out


# In[76]:


class Encodings(nn.Module):
    def __init__(self,device='cuda',dtype=torch.float32,token_size=1024,d_model=256):
        super().__init__()
        self.positional_encoding1=nn.Parameter(torch.randn((token_size,d_model),device=device,dtype=dtype))
        self.positional_encoding2=nn.Parameter(torch.randn((token_size,d_model),device=device,dtype=dtype))
    def forward(self,optical,sar):
        return (optical+self.positional_encoding1),(sar+self.positional_encoding2)



# In[77]:


class CrossAttention(nn.Module):
    def __init__(self,token_size=1024,d_model=256,head_dim=32,device='cuda',dtype=torch.float32):
        super().__init__()
        self.token_size=token_size
        self.d_model=d_model
        self.head_dim=head_dim
        self.wq=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wk=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wv=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wout=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.layernorm1=nn.LayerNorm(256,device=device,dtype=dtype)
        self.layernorm2=nn.LayerNorm(256,device=device,dtype=dtype)
    def forward(self,optical,sar):
        B,S,D=optical.shape
        OPTICAL=self.layernorm1(optical)
        SAR=self.layernorm2(sar)
        q=self.wq(OPTICAL)
        k=self.wk(SAR)
        v=self.wv(SAR)
        q=q.reshape(B,S,-1,self.head_dim)
        q=q.transpose(1,2)
        k=k.reshape(B,S,-1,self.head_dim)
        k=k.transpose(1,2)
        v=v.reshape(B,S,-1,self.head_dim)
        v=v.transpose(1,2)
        out=f.scaled_dot_product_attention(q,k,v,is_causal=False)
        out=out.transpose(1,2)
        out=out.reshape(B,S,-1)
        out=self.wout(out)
        return out+optical # residual implemented 







# In[78]:


class SelfAttention(nn.Module):
    def __init__(self,token_size=1024,d_model=256,head_dim=32,device='cuda',dtype=torch.float32):
        super().__init__()
        self.token_size=token_size
        self.d_model=d_model
        self.head_dim=head_dim
        self.wq=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wk=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wv=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.wout=nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.layernorm=nn.LayerNorm(256,device=device,dtype=dtype)
    def forward(self,optical):
        B,S,D=optical.shape
        OPTICAL=self.layernorm(optical)
        q=self.wq(OPTICAL)
        k=self.wk(OPTICAL)
        v=self.wv(OPTICAL)
        q=q.reshape(B,S,-1,self.head_dim)
        q=q.transpose(1,2)
        k=k.reshape(B,S,-1,self.head_dim)
        k=k.transpose(1,2)
        v=v.reshape(B,S,-1,self.head_dim)
        v=v.transpose(1,2)
        out=f.scaled_dot_product_attention(q,k,v,is_causal=False)
        out=out.transpose(1,2)
        out=out.reshape(B,S,-1)
        out=self.wout(out)
        return out+optical # residual implemented



# In[79]:


class FNN(nn.Module):
    def __init__(self,device='cuda',dtype=torch.float32,fnn_size=1024):
        super().__init__()
        self.up=nn.Linear(256,fnn_size,device=device,dtype=dtype)
        self.down=nn.Linear(fnn_size,256,device=device,dtype=dtype)
        self.layernorm=nn.LayerNorm(256,device=device,dtype=dtype)
    def forward(self,x):
        X=self.layernorm(x)
        X=f.gelu(self.up(X))
        X=self.down(X)
        return X+x



# In[80]:


class BLOCK(nn.Module):
    def __init__(self,CrossAttention,SelfAttention,FNN):
        super().__init__()
        self.crossattention=CrossAttention()
        self.selfattention=SelfAttention()
        self.fnn=FNN()

    def forward(self,optical,sar):
        Y=self.crossattention(optical,sar)
        Y=self.selfattention(Y)
        Y=self.fnn(Y)
        return Y


# In[81]:


class Decoder(nn.Module):
    def __init__(self,ConvBlock,In_channels,Out_channels1,Out_channels2,Out_channels3,skip1_size,skip2_size,skip3_size,Kernel_size,Stride,Padding,device,dtype):
        super().__init__()
        self.ConvBlock1=ConvBlock(In_channels+skip3_size,Out_channels1,Kernel_size,Stride,Padding,device,dtype)
        self.ConvBlock2=ConvBlock(Out_channels1+skip2_size,Out_channels2,Kernel_size,Stride,Padding,device,dtype)
        self.ConvBlock3=ConvBlock(Out_channels2+skip1_size,Out_channels3,Kernel_size,Stride,Padding,device,dtype)

    def forward(self,x,skip3,skip2,skip1):
        x=f.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x=torch.cat((x,skip3),dim=1)
        x=self.ConvBlock1(x)

        x=f.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x=torch.cat((x,skip2),dim=1)
        x=self.ConvBlock2(x)

        x=f.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        x=torch.cat((x,skip1),dim=1)
        x=self.ConvBlock3(x)


        return x




# In[82]:


class Model(nn.Module):
    def __init__(self,ConvBlock,Encoder,Encodings,CrossAttention,SelfAttention,FNN,BLOCK,Decoder,eI1=3,eI2=2,eO1=64,eO2=128,eO3=256,dI=512,dO1=256,dO2=128,dO3=64,s1=64,s2=128,s3=256,Kernel_size=3,Stride=1,Padding=1,token_size=1024,d_model=256,head_dim=32,device='cuda',dtype=torch.float32):
        super().__init__()
        self.encoder_optical=Encoder(ConvBlock,eI1,eO1,eO2,eO3,Kernel_size,Stride,Padding,device,dtype)
        self.encoder_sar=Encoder(ConvBlock,eI2,eO1,eO2,eO3,Kernel_size,Stride,Padding,device,dtype)
        self.encodings=Encodings()
        self.block1=BLOCK(CrossAttention,SelfAttention,FNN)
        self.block2=BLOCK(CrossAttention,SelfAttention,FNN)
        self.block3=BLOCK(CrossAttention,SelfAttention,FNN)
        self.convblock1=ConvBlock(eO3,dI,Kernel_size,Stride,Padding,device,dtype)
        self.convblock2=ConvBlock(dI,dI,Kernel_size,Stride,Padding,device,dtype)
        self.decoder=Decoder(ConvBlock,dI,dO1,dO2,dO3,s1,s2,s3,Kernel_size,Stride,Padding,device,dtype)
        self.convblock3=nn.Conv2d(dO3,3,kernel_size=1,device='cuda',dtype=torch.float32)

    def forward(self,optical,sar):
        B=optical.shape[0]
        skip1,skip2,skip3,optical=self.encoder_optical(optical)
        a,b,c,sar=self.encoder_sar(sar)

        optical_flatten=optical.reshape(B,-1,1024).transpose(1,2)
        sar_flatten=sar.reshape(B,-1,1024).transpose(1,2)
        optical_flatten,sar_flatten=self.encodings(optical_flatten,sar_flatten)
        Y=self.block1(optical_flatten,sar_flatten)
        Y=self.block2(Y,sar_flatten)
        Y=self.block3(Y,sar_flatten)
        Y=Y.transpose(1,2)
        Y=Y.reshape(B,-1,32,32)
        decoder_input=self.convblock1(Y)
        decoder_input=self.convblock2(decoder_input)
        decoder_output=self.decoder(decoder_input,skip3,skip2,skip1)
        output=self.convblock3(decoder_output)
        return torch.sigmoid(output)







# In[83]:


from pytorch_msssim import ssim


# In[84]:


def weighted_l1(mask,pred,target,w=5.0):
    weight=1+ (w-1)*mask
    return (weight*(pred-target).abs()).mean()


# In[85]:


def spectral_loss(mask, pred, target, eps=1e-6):
    m = mask.expand_as(pred)
    n = m.sum(dim=(2,3)).clamp(min=1)                         # [B,3]
    mp = (pred*m).sum(dim=(2,3)) / n
    mt = (target*m).sum(dim=(2,3)) / n
    vp = (((pred   - mp[..., None, None])**2) * m).sum(dim=(2,3)) / n
    vt = (((target - mt[..., None, None])**2) * m).sum(dim=(2,3)) / n
    sp, st = (vp + eps).sqrt(), (vt + eps).sqrt()
    return (mp - mt).abs().mean() + (sp - st).abs().mean()


# In[86]:


class Train:
    def __init__(self,model,optimizer,scheduler):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.step=0
    def __call__(self,optical_input_batch,sar_input_batch,target_batch,mask_batch):
        self.optimizer.zero_grad()
        L=0
        for i in range(4):
            optical=optical_input_batch[i,:,:,:,:]
            
            sar=sar_input_batch[i,:,:,:,:]
            mask=mask_batch[i,:,:,:,:]
            target=target_batch[i,:,:,:,:]
            pred=self.model(optical,sar)
            l1=weighted_l1(mask,pred,target)
            ssim_loss=1-ssim(pred,target,data_range=1.0,size_average=True)
            spectral=spectral_loss(mask,pred,target)
            loss=(l1 + 0.5*(ssim_loss) + 0.05*spectral)/4
            loss.backward()
            L+=loss
        gn=nn.utils.clip_grad_norm_(self.model.parameters(),1.0)    
        self.optimizer.step()
        self.scheduler.step()
        self.step+=1   
        return L,gn.item()
    def save(self,name='checkpoint1'):
        ckpt={'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'steps':self.step}
        torch.save(ckpt,name)

    def load(self,name='checkpoint1'):
        ckpt=torch.load(name)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.step=ckpt['steps']


# In[ ]:





# In[ ]:




