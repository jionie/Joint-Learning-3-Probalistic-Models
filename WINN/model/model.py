import numpy as np
import os
import sys
import PIL
import time
import copy
import scipy
import sklearn
import math

from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from torch import autograd
from torch.autograd import Variable, grad

from sklearn.utils import shuffle

from utils.data_io import *

from WINN_utils import *

from opts import *

import random

import cv2

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # pdb.set_trace()
        # return self.gamma * (x - mean) / (std + self.eps) + self.beta
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        # print ("to x {}".format(x.data.numpy().shape))
        # print ("to gamma {}".format(self.gamma.shape))
        # print ("to beta {}".format(self.beta.shape))
        # print ("to mean {}".format(mean.data.numpy().shape))
        # print ("to std {}".format(std.data.numpy().shape))

        y = (x - mean) / (std + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

#"same padding" padding = int((kernel_size-1)/2)
class ConvPadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(ConvPadding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1)/2)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, \
                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, \
                            dilation=self.dilation, groups=self.groups, bias=self.bias)
        
    def forward(self, x):
        out = self.conv(x)
        
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvMeanPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = ConvPadding(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, self.bias)
        
    def forward(self, x):
        conv = self.conv(x)
        
        #::k ervery k element, s=range(20), s[::3]=[0, 3, 6, 9, 12, 15, 18]
        #L[x::y] means a slice of L where the x is the index to start from and y is the step size.
        #here we 
        out = (conv[:,:,::2,::2] + conv[:,:,1::2,::2] + conv[:,:,::2,1::2] + conv[:,:,1::2,1::2])/4.
        
        return out

class MeanPoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MeanPoolConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = ConvPadding(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, self.bias)
        
    def forward(self, x):
        
        x = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2])/4.
        out = self.conv(x)
        
        return out

#Rearranges data from depth into blocks of spatial data. This is the reverse transformation of SpaceToDepth
#This operation is useful for resizing the activations between convolutions (but keeping all data), 
#e.g. instead of pooling. It is also useful for training purely convolutional models.
#N H W C
#Chunks of data of size block_size * block_size from depth 
#are rearranged into non-overlapping blocks of size block_size x block_size
#The width the output tensor is input_depth * block_size, whereas the height is input_height * block_siz
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1) #N H W C
        (batch_size, in_height, in_width, in_channels) = x.size()
        out_channels = int(in_channels / self.block_size / self.block_size)
        out_width = int(in_width * self.block_size)
        out_height = int(in_height * self.block_size)
        out = x.reshape(batch_size, in_height, in_width, self.block_size*self.block_size, out_channels)
        #N H W BLOCK*BLOCK C/BLOCK/BLOCK
        
        splits = out.split(self.block_size, dim=3)
        #BLOCK (N H W BLOCK C/BLOCK/BLOCK) list
        
        #If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible). 
        #Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

        #If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks 
        #with sizes in dim according to split_size_or_sections.
        
        #split -> (N H W BLOCK C/BLOCK/BLOCK) -> reshape (N H W*BLOCK C/BLOCK/BLOCK)
        #stacks -> BLOCK (N H W*BLOCK C/BLOCK/BLOCK)
        stacks = [split.reshape(batch_size, in_height, out_width, out_channels) for split in splits]
        #stacks -> BLOCK N H W*BLOCK C/BLOCK/BLOCK
        stacks = torch.stack(stacks, 0)
        #stacks -> N BLOCK H W*BLOCK C/BLOCK/BLOCK
        stacks = stacks.transpose(0, 1)
        #stacks -> N H*BLOCK W*BLOCK C/BLOCK/BLOCK
        stacks = stacks.reshape(batch_size, out_height, out_width, out_channels)
        #out -> N C/BLOCK/BLOCK H*BLOCK W*BLOCK
        out = stacks.permute(0, 3, 1, 2)
        
        return out       

class UpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(UpsamplingConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.depth_to_space = DepthToSpace(2)
        self.conv = ConvPadding(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, self.bias)
        
    def forward(self, x):
        
        x = torch.cat((x, x, x, x), 1)
        #x -> N H W C*4
        x = x.permute(0, 2, 3, 1)
        #x -> N C H*2 W*2
        x = self.depth_to_space(x)
        out = self.conv(x)
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, resample=None):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        if resample == 'down':
            self.bn1 = LayerNorm(in_channels)
            self.bn2 = LayerNorm(in_channels)
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif resample == None:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = LayerNorm(out_channels)
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            
            self.conv_shortcut = MeanPoolConv(self.in_channels, self.out_channels, 1, self.stride, 
                                             self.padding, self.dilation, self.groups, self.bias)
            
            self.conv_1 = ConvPadding(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, bias=False)
            
            self.conv_2 = ConvMeanPool(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                             self.padding, self.dilation, self.groups, self.bias)
        
        elif resample == 'up':
            
            self.conv_shortcut = UpsamplingConv(self.in_channels, self.out_channels, 1, self.stride, 
                                             self.padding, self.dilation, self.groups, self.bias)
            
            self.conv_1 = UpsamplingConv(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                             self.padding, self.dilation, self.groups, bias=False)
            
            self.conv_2 = ConvPadding(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, self.bias)
        
        elif resample == None:
            
            self.conv_shortcut = ConvPadding(self.in_channels, self.out_channels, 1, self.stride, self.dilation, 
                        self.groups, self.bias)
            
            self.conv_1 = ConvPadding(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, bias=False)
            
            self.conv_2 = ConvPadding(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.dilation, 
                        self.groups, self.bias)
            
        else:
            
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class Good_Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Good_Discriminator, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        
        self.conv1 = ConvPadding(self.in_channels, dim, 3, 1)
        self.resblock1 = ResidualBlock(self.dim, 2*self.dim, 3, 'down')
        self.resblock2 = ResidualBlock(2*self.dim, 4*self.dim, 3, 'down')
        self.resblock3 = ResidualBlock(4*self.dim, 8*self.dim, 3, 'down')
        self.resblock4 = ResidualBlock(8*self.dim, 8*self.dim, 3, 'down')
        
        self.linear = nn.Linear(4*4*8*self.dim, 1)
        
    def forward(self, x, img_size=64):
        
        x = x.contiguous()
        #N H W C -> N C H W
        x = x.permute(0, 3, 1, 2)
       
        x = x.view(-1, self.in_channels, img_size, img_size)
        
        conv1 = self.conv1(x)
        #no activation, why???
        
        res1 = self.resblock1(conv1)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        
        out = res4.view(-1, 4*4*8*self.dim)
        out = self.linear(out)
        out = out.view(-1)
        
        return out


class INN_Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64, batch_norm=True):
        super(INN_Discriminator, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.batch_norm = batch_norm
        
        self.conv1 = ConvPadding(self.in_channels, int(dim/2), 3, 1)
        self.swish = swish()
        
        self.conv2 = ConvPadding(int(dim/2), dim, 3, 1)
        self.layernorm1 = LayerNorm(dim)
        
        self.meanpoolconv1 = MeanPoolConv(dim, dim, 3, 1)
        self.layernorm2 = LayerNorm(dim)
        
        self.conv3 = ConvPadding(dim, dim*2, 3, 1)
        self.layernorm3 = LayerNorm(dim*2)
        
        self.meanpoolconv2 = MeanPoolConv(dim*2, dim*2, 3, 1)
        self.layernorm4 = LayerNorm(dim*2)
        
        self.conv4 = ConvPadding(dim*2, dim*4, 3, 1)
        self.layernorm5 = LayerNorm(dim*4)
        
        self.meanpoolconv3 = MeanPoolConv(dim*4, dim*4, 3, 1)
        self.layernorm6 = LayerNorm(dim*4)
        
        self.conv5 = ConvPadding(dim*4, dim*8, 3, 1)
        self.layernorm7 = LayerNorm(dim*8)
        
        self.linear = nn.Linear(4*4*8*dim, 1)
        
        
    def forward(self, x, img_size=64):
        
        x = x.contiguous()
        x = x.view(-1, self.in_channels, img_size, img_size)
        
        conv1 = self.swish(self.conv1(x))
        #N self.dim/2 img_size img_size
        
        if(self.batch_norm):
            conv2 = self.swish(self.layernorm1(self.conv2(conv1)))
            #N self.dim img_size img_size
        else:
            conv2 = self.swish(self.conv2(conv1))
            #N self.dim img_size img_size
            
        if(self.batch_norm):
            meanpoolconv1 = self.swish(self.layernorm2(self.meanpoolconv1(conv2)))
            #N self.dim img_size/2 img_size/2
        else:
            meanpoolconv1 = self.swish(self.meanpoolconv1(conv2))
            #N self.dim img_size/2 img_size/2
        
        if(self.batch_norm):
            conv3 = self.swish(self.layernorm3(self.conv3(meanpoolconv1)))
            #N self.dim*2 img_size/2 img_size/2
        else:
            conv3 = self.swish(self.conv3(meanpoolconv1))
            #N self.dim*2 img_size/2 img_size/2
        
        if(self.batch_norm):
            meanpoolconv2 = self.swish(self.layernorm4(self.meanpoolconv2(conv3)))
            #N self.dim*2 img_size/4 img_size/4
        else:
            meanpoolconv2 = self.swish(self.meanpoolconv2(conv3))
            #N self.dim*2 img_size/4 img_size/4
        
        if(self.batch_norm):
            conv4 = self.swish(self.layernorm5(self.conv4(meanpoolconv2)))
            #N self.dim*4 img_size/4 img_size/4
        else:
            conv4 = self.swish(self.conv4(meanpoolconv2))
            #N self.dim*4 img_size/4 img_size/4
        
        if(self.batch_norm):
            meanpoolconv3 = self.swish(self.layernorm6(self.meanpoolconv3(conv4)))
            #N self.dim*4 img_size/8 img_size/8
        else:
            meanpoolconv3 = self.swish(self.meanpoolconv3(conv4))
            #N self.dim*4 img_size/8 img_size/8
        
        if(self.batch_norm):
            conv5 = self.swish(self.layernorm7(self.conv5(meanpoolconv3)))
            #N self.dim*8 img_size/8 img_size/8 
        else:
            conv5 = self.swish(self.conv5(meanpoolconv3))
            #N self.dim*8 img_size/8 img_size/8 
        
        out = (conv5[:, :, ::2, ::2] + conv5[:, :, 1::2, ::2] + conv5[:, :, ::2, 1::2] + conv5[:, :, 1::2, 1::2])/4
        #N self.dim*8 img_size/16 img_size/16 
        
        out = out.view(-1, int(img_size/16)*int(img_size/16)*8*self.dim)
        
        out = self.linear(out)
        #N 1
        
        out = out.view(-1)
        #N
        
        return out

#Noise samples, initial psudo negatives
class Noise(nn.Module):
    def __init__(self, n_samples, dim = 64):
        super(Noise, self).__init__()
        
        self.n_samples = n_samples
        self.dim = dim
        self.conv1 = ConvPadding(8*self.dim, 4*self.dim, 5, 1)
        self.upsample1 = nn.Upsample(size=8, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm1 = LayerNorm(4*self.dim)
        
        self.conv2 = ConvPadding(4*self.dim, 2*self.dim, 5, 1)
        self.upsample2 = nn.Upsample(size=16, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm2 = LayerNorm(2*self.dim)
        
        self.conv3 = ConvPadding(2*self.dim, self.dim, 5, 1)
        self.upsample3 = nn.Upsample(size=32, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm3 = LayerNorm(self.dim)
        
        self.conv4 = ConvPadding(self.dim, 3, 5, 1)
        self.upsample4 = nn.Upsample(size=64, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm4 = LayerNorm(3)
        
    def forward(self, x):
        
        conv1 = self.conv1(x)
        upsample1 = self.upsample1(conv1)
        upsample1 = self.layernorm1(upsample1)
        #N 4*dim 8 8
        
        conv2 = self.conv2(upsample1)
        upsample2 = self.upsample2(conv2)
        upsample2 = self.layernorm2(upsample2)
        #N 2*dim 16 16
        
        conv3 = self.conv3(upsample2)
        upsample3 = self.upsample3(conv3)
        upsample3 = self.layernorm3(upsample3)
        #N dim 32 32
        
        conv4 = self.conv4(upsample3)
        upsample4 = self.upsample4(conv4)
        upsample4 = self.layernorm4(upsample4)
        #N dim/2 64 64
        
        return upsample4

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index
    def __len__(self):
        return len(self.dataset)
      

class WINN(nn.Module):
    def __init__(self, opts, in_channels=3, dim=64, cascades=4, iterations_per_cascade=100, discriminator_train_steps=3, LAMBDA=10):
        super(WINN, self).__init__()
        self.batch_size = opts.batch_size
        self.in_channels = in_channels

        if(opts.set=='cifar'):
            opts.img_size = 32
            print("training on cifar with image size: %i" %(opts.img_size))

        self.img_size = opts.img_size
        self.dim = dim
        self.num_chain = opts.nRow*opts.nCol #each image in final result
        self.opts = opts
        self.Noise_Provider = Noise(100, dim=64)
        self.cascades = cascades
        self.iterations_per_cascade = iterations_per_cascade
        self.discriminator_train_steps = discriminator_train_steps
        self.LAMBDA = LAMBDA


    def calc_gradient_penalty(self, real_data, fake_data):

        alpha = torch.rand(int(self.batch_size/2), 1)
        alpha = alpha.expand(int(self.batch_size/2), int(real_data.nelement()/int(self.batch_size/2))).contiguous().view(int(self.batch_size/2), \
            self.in_channels, self.dim, self.dim)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA

        return gradient_penalty
    
            
    def langevin_dynamics_discriminator(self, x):
        
        #run langevin_step_num_gen steps langevin dynamics
            
        #dimension of x is 3
        noise = Variable(torch.randn(self.batch_size, 3, self.opts.img_size, self.opts.img_size).cuda())
        #"However, .data can be unsafe in some cases. 
        #Any changes on x.data wouldn’t be tracked by autograd, 
        #and the computed gradients would be idiscriminator_train_steporrect if x is needed in a backward pass. 
        #A safer alternative is to use x.detach(), 
        #which also returns a Tensor that shares data with requires_grad=False, 
        #but will have its in-place changes reported by autograd if x is needed in backward."
        
        # clone it and turn x into a leaf variable so the grad won't be thrown away
        x = Variable(x.data, requires_grad=True)

        S_optimizer = torch.optim.Adam([x], lr=0.01, betas=[self.opts.beta1_dis, 0.9])
        
        x_feature = self.discriminator(x)
        #x_feature is f(x;\theta) which is \ln(p(y=1|x,\theta)/p(y=0|x,\theta))

        loss = x_feature.mean()
        
        #torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, 
        #create_graph=False, grad_variables=None)
        
        loss.backward(-1*torch.FloatTensor([1]).cuda())
        
        #grad = \frac{\partial f(x;\theta)}{\partial x}
        #grad = x.grad
        
        # print ('x is : '+str(x[0]))
        # print ('x_grad is : '+str(grad[0]))

        S_optimizer.step()
        
        #x = x + 0.5 * self.opts.langevin_step_size_dis * self.opts.langevin_step_size_dis * grad
        
        #+ step_size*U_{\tau}
        if self.opts.with_noise:
            x += self.opts.langevin_step_size_dis * noise
                
        return x, loss
        
    def train(self, discriminator_model=None, LAMBDA=10.0):
        
        start_time = time.time()
        
        if not os.path.exists(self.opts.ckpt_dir):
            os.makedirs(self.opts.ckpt_dir)
        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)
        logfile = open(self.opts.ckpt_dir + '/log', 'w+')
        
        # Prepare for root directory of intermediate image.
        intermediate_image_root = os.path.join(self.opts.output_dir, "intermediate")
        # Prepare for root directory of negative images.
        neg_image_root = os.path.join(self.opts.output_dir, "negative")


        if not os.path.exists(intermediate_image_root):
            os.makedirs(intermediate_image_root)
        
        ######################################################################
        # Training stage 1: Load positive images.
        ######################################################################
        print("Training stage 1: Load positive images...", file=logfile)

        if self.opts.set == 'scene' or self.opts.set == 'cifar':
            train_data = DataSet(os.path.join(self.opts.data_path, self.opts.category), 
                                 image_size=self.opts.img_size)
        else:
            train_data = torchvision.datasets.LSUN(root=self.opts.data_path,
                                                   classes=['bedroom_train'],
                                                   transform=transforms.Compose([transforms.Resize(self.img_size),
                                                   transforms.ToTensor(), ]))
            
        num_batches = int(math.ceil(len(train_data) / self.batch_size))

        self.Noise_Provider.apply(init_weights).eval().cuda()
        neg_image_path = os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png')
        neg_init_images_count = 10000
        neg_init_images_path = [neg_image_path.format(0, 0, i) for i in range(neg_init_images_count)]


        if not os.path.exists(neg_image_root):
            os.makedirs(neg_image_root)
            # Generate initial pseudo-negative images
            # In fact, the name of image has format 
            #     {cascade}_{next iteration}_{i}.png
            # where cascade means current cascade model, next iteration means
            # next iteration of sampler and discriminator training, and i means
            # the index of images.
            S_iteration_count_of_batch = neg_init_images_count // self.batch_size
            
            for i in range(S_iteration_count_of_batch):             
                small_noise_batch = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, 512, 4, 4))
                small_noise_batch_cuda = torch.tensor(small_noise_batch, dtype=torch.float32).cuda()
                big_noise_batch = self.Noise_Provider(small_noise_batch_cuda).detach().cpu().numpy()
                np_noise_images = np.transpose(big_noise_batch, axes=[0, 2, 3, 1])

                # Generate random images as negatives and save them.
                for j in range(100):#, neg_init_image_path in enumerate(neg_init_images_path):
                    # Attention: Though it is called neg_image here, it has 4 dimensions,
                    #            that is, [1, height, width, channels], which is not a
                    #            pure single image, which is [height, width, channels].
                    #            So we still use save_unnormalized_images here instead of 
                    #            save_unnormalized_image.
                    neg_image = np_noise_images[j].reshape(1, 64, 64, 3)
                    neg_image = neg_image - neg_image.min()
                    neg_image = neg_image / neg_image.max() * 255.0 
                    save_unnormalized_images(images = neg_image, 
                                            size = (1, 1), path = neg_init_images_path[self.batch_size * i + j])

        neg_all_images_path = neg_init_images_path
        image_shape = train_data.images.shape[1:]
        self.Noise_Provider.cpu()
        torch.cuda.empty_cache()
        ######################################################
        print("Positive images {0}, negative images {1}, image shape {2}"\
            .format(train_data.images.shape[0], len(neg_all_images_path), image_shape), file=logfile)
        
        ######################################################################
        # Training stage 3: Cascades training.
        ######################################################################
        print("Training stage 2: Cascades training...", file=logfile)

        # Prepare for the initial images to feed the sampler. In fact, it is 
        # because we always use negative images in last cascade as the "initial"
        # images to feed sampler in all iterations of current cascade.
        S_neg_last_cascade_images_path = copy.deepcopy(neg_all_images_path)
        half_batch_size = int(self.batch_size/2) ## half of positive images, half of negative images
        

        for cascade in range(self.cascades):
        ######################################################################
        # Training stage 3.1: Iterations training.
        ######################################################################
        # One iteration means one time of discriminator training and one time
        # of sampling pseudo-negatives. One iteration training may contain multiple
        # batches for discriminator training and sampling pseudo-negatives.
            
            if(discriminator_model!=None):
                self.discriminator = torch.load(discriminator_model).train().cuda()
                print('Loading Discriminator from ' + discriminator_model + '...')
            else:
                self.discriminator = INN_Discriminator().apply(init_weights).train().cuda()
                print('Loading Discriminator without initialization...')
            
            if(self.opts.with_noise):
                print("Langevin Dynamics with noise")
            else:
                print("Langevin Dynamics without noise")
            
            for p in self.discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.opts.lr_dis,
                                         betas=[self.opts.beta1_dis, 0.9])

            for iteration in range(self.iterations_per_cascade):
                ######################################################################
                # Training stage 3.1.1: Prepare images and labels for discriminator
                # training.
                ######################################################################
                # Count of positive images to train in current iteration.
                D_pos_iteration_images_count = int(min(iteration + 1, 5) * 1000 \
                    // half_batch_size * half_batch_size)
                
                if D_pos_iteration_images_count >= len(train_data):
                    # When the number of all positive images is more than current
                    # iteration negative images, we allow duplicate images.
                    D_pos_iteration_images = np.repeat(train_data, int(D_pos_iteration_images_count/len(train_data))+1, axis=0)
                    #print(D_pos_iteration_images.shape)
                    np.random.shuffle(D_pos_iteration_images)
                    #print(D_pos_iteration_images.shape)
                    D_pos_iteration_images = D_pos_iteration_images[:D_pos_iteration_images_count]
                    #print(D_pos_iteration_images.shape)

                else:
                    # When the number of all positive images is less or equal than 
                    # current iteration negative images, we require unique images.
                    train_data.shuffle()
                    D_pos_iteration_images = train_data[:D_pos_iteration_images_count]
                    #print(D_pos_iteration_images.shape)

                
                # Here we consider the "save all" mode in Long Jin's code. This mode
                # has different behaviors on discriminator and sampler.
                # 1) Discriminator.
                #     We draw positive images from training dataset and the same
                #     number of negative images from *all* pseudo-negative images in data/negative folder.
                #     Every iteration of sampler will add newly generated negative images
                #     into all data/negative foler.
                # 2) Sampler.
                #     We draw "initial" negative images in every iterations in current
                #     cascade from part of generated negative images in last cascade.
                #     More specificially, the part is the *last* iteration of last cascade.

                #for i in range(len(D_pos_iteration_images)):
                    #img = D_pos_iteration_images[i].transpose(1, 2, 0)
                    #cv2.imshow('image',img)
                    #cv2.waitKey(0)

                D_neg_iteration_images_count = D_pos_iteration_images_count
                train_data_negatives = DataSet(data_list=neg_all_images_path,image_size=self.opts.img_size)
                train_data_negatives.shuffle()
                D_neg_iteration_images = train_data_negatives[:D_neg_iteration_images_count]
                del train_data_negatives
                
                #for i in range(len(D_neg_iteration_images)):
                    #img = D_neg_iteration_images[i].transpose(1, 2, 0)
                    #cv2.imshow('image',img)
                    #cv2.waitKey(0)


                D_learning_rate = self.opts.lr_dis

                print(("Discriminator: Cascade {0}, iteration {1}, " + 
                   "all pos {2}, all neg {3}, " + 
                   "current iteration {4} (pos {5}, neg {6}), " + 
                   "learning rate {7}").format(\
                    cascade, iteration, \
                    D_pos_iteration_images.shape[0], D_neg_iteration_images.shape[0], \
                    D_pos_iteration_images_count + D_neg_iteration_images_count, \
                    D_pos_iteration_images_count, D_neg_iteration_images_count, \
                    D_learning_rate), file=logfile)

                ######################################################################
                # Training stage 3.1.2: Train the discriminator.
                ######################################################################
                # Count of batch in discriminator training in current iteration. 
                D_iteration_count_of_batch = int(D_pos_iteration_images.shape[0] // half_batch_size)
                
                min_D_batch_pos_loss = np.inf
                max_D_batch_pos_loss = -np.inf

                #summary(self.discriminator, (3, 64, 64))

                # Training for self.discriminator_train_steps steps
                for discriminator_train_step in range(self.discriminator_train_steps):
                    for i in range(D_iteration_count_of_batch):

                        one = torch.FloatTensor([1]).cuda()
                        mone = (one * -1)

                        self.discriminator.zero_grad()

                        #img = D_neg_iteration_images[0].transpose(1, 2, 0)
                        #cv2.imshow('image',img)
                        #cv2.waitKey(0)

                        #img = D_pos_iteration_images[0].transpose(1, 2, 0)
                        #cv2.imshow('image',img)
                        #cv2.waitKey(0)

                        # Load images for this batch in discriminator.
                        D_pos_batch_images = D_pos_iteration_images[i * half_batch_size : (i + 1) * half_batch_size, :, :, :]
                        D_neg_batch_images = D_neg_iteration_images[i * half_batch_size : (i + 1) * half_batch_size, :, :, :]
                        #print(D_pos_batch_images.shape)

                        D_pos_batch_images_cuda = torch.tensor(D_pos_batch_images, dtype=torch.float).cuda()
                        D_neg_batch_images_cuda = torch.tensor(D_neg_batch_images, dtype=torch.float).cuda()

                        #wgan gp
                        
                        D_pos_batch_images_cuda_v = autograd.Variable(D_pos_batch_images_cuda)
                        D_pos_batch_logits = self.discriminator(D_pos_batch_images_cuda_v)
                        D_pos_batch_logits = D_pos_batch_logits.mean()
                        D_pos_batch_logits.backward(mone)


                        D_neg_batch_images_cuda_v = autograd.Variable(D_neg_batch_images_cuda)
                        D_neg_batch_logits = self.discriminator(D_neg_batch_images_cuda_v)
                        D_neg_batch_logits = D_neg_batch_logits.mean()
                        D_neg_batch_logits.backward(one)

                        gradient_penalty = self.calc_gradient_penalty(D_pos_batch_images_cuda_v.data, D_neg_batch_images_cuda_v)
                        gradient_penalty.backward()

                        dis_optimizer.step()

                        D_loss = D_neg_batch_logits - D_pos_batch_logits + gradient_penalty
                        Wasserstein_D = D_pos_batch_logits - D_neg_batch_logits

                        
                        #print("true logots: ", sigmoid(D_pos_batch_logits.detach().cpu().numpy()), "fake logots: ", sigmoid(D_neg_batch_logits.detach().cpu().numpy()))
                        print("Discriminator Loss: ", D_loss.detach().cpu().numpy(), "Wasserstein Distance: ", Wasserstein_D.detach().cpu().numpy())
                        #torch.cuda.empty_cache()

                        if (discriminator_train_step == self.discriminator_train_steps - 1):
                            # Positive samples' loss after training in current iteration.
                            # It will be used as an early stopping threshold when we generate pseudo-negative samples
                            D_batch_pos_loss = D_pos_batch_logits.detach().cpu().numpy()
                            if (D_batch_pos_loss < min_D_batch_pos_loss):
                                min_D_batch_pos_loss = D_batch_pos_loss
                            if (D_batch_pos_loss > max_D_batch_pos_loss):
                                max_D_batch_pos_loss = D_batch_pos_loss

                        print("Discriminator: Cascade {0}, iteration {1}, Critic {2}, time {3}, D_loss {4}, D_pos_loss {5}, {6}"\
                            .format(cascade, iteration, discriminator_train_step, time.time() - start_time, D_loss.detach().cpu().numpy(), \
                            min_D_batch_pos_loss, max_D_batch_pos_loss), file=logfile)

                
                # Save last batch images in discriminator training.
                D_intermediate_image_path = os.path.join(intermediate_image_root, 'D_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
                sqrt_batch_size = int(np.sqrt(self.batch_size))
                D_intermediate_images = np.transpose(np.concatenate((D_pos_batch_images, D_neg_batch_images), axis=0), (0, 2, 3, 1))
                save_unnormalized_images(images = unnormalize(D_intermediate_images), \
                                        size = (sqrt_batch_size, sqrt_batch_size), \
                                        path = D_intermediate_image_path)        
                        

                ######################################################################
                # Training stage 3.1.3: Initialize pseudo-negatives.
                ######################################################################

                # Load path of negative images in last cascade and shuffle.
                S_neg_last_cascade_images_path = shuffle(S_neg_last_cascade_images_path)
                # Attention again, the last cascade here does not mean all negative images
                # produced in last cascade, but only negative images in last iteration of
                # last cascade. Initial negatives

                # Number of negative images to be generated in current iteration.
                if iteration == self.iterations_per_cascade - 1:
                    # Generate more in last iteration of cascade.
                    S_neg_iteration_images_count = 10000
                else:
                    S_neg_iteration_images_count = 1000
                
                S_neg_current_iteration_images_path = \
                    [os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png')\
                    .format(cascade, iteration + 1, i) for i in range(S_neg_iteration_images_count)]
                
                print(("Sampler: Cascade {0}, iteration {1}, " + 
                    "current iteration neg {2}").format(
                    cascade, iteration, 
                    S_neg_iteration_images_count), file=logfile)

                # Saved early-stopping threshold in the model, max_D_batch_pos_loss and min_D_batch_pos_loss

                ######################################################################
                # Training stage 3.1.3: Sample pseudo-negatives.
                ######################################################################
                # Count of batch  in current iteration. 
                S_iteration_count_of_batch = int(S_neg_iteration_images_count // self.batch_size)
                psudo_neg_images = DataSet(data_list=S_neg_last_cascade_images_path,image_size=self.opts.img_size)
                psudo_neg_images.shuffle()

                for i in range(S_iteration_count_of_batch):

                    # Load images from last cascade generated negative images.
                    # We mention it again, that is, in each iteration in current cascade, 
                    # we will generate images based on last cascade, but not last iteration. 
                    # It is quite a strange strategy.
                    
                    S_neg_batch_images = psudo_neg_images[i * self.batch_size : (i + 1) * self.batch_size]
                    S_neg_batch_images_cuda = torch.tensor(S_neg_batch_images, dtype=torch.float).cuda()

                    thres_ = np.random.uniform(min_D_batch_pos_loss, max_D_batch_pos_loss)

                    print("maximum langevin_step_num_dis: ", self.opts.langevin_step_num_dis)
                    print("Sampling: " + str(i) + " round of " + str(S_iteration_count_of_batch) + " total rounds")

                    S_neg_batch_images_v = Variable(S_neg_batch_images_cuda, requires_grad=True)
                    for j in range(self.opts.langevin_step_num_dis):
                        # Optimize.
                        S_neg_batch_images_v, S_loss = self.langevin_dynamics_discriminator(S_neg_batch_images_v)
                        S_neg_batch_images_v = Variable(S_neg_batch_images_v, requires_grad=True)
                        S_loss = S_loss.detach().cpu().numpy()
                        # Stop based on threshold.
                        # The threshold is based on real samples' score.
                        # Update until the WINN network thinks pseudo-negative samples are quite close to real.

                        if(j%100==0):
                            print("step: ", j, "S_loss: ", S_loss, "min_pos_loss: ", min_D_batch_pos_loss, "max_pos_loss: ", max_D_batch_pos_loss, \
                                "thres: ", thres_)

                        if S_loss >= thres_:
                            break
                    
                    S_neg_intermediate_images = S_neg_batch_images_v.detach().cpu().numpy()
                    S_neg_intermediate_images = np.transpose(S_neg_intermediate_images, (0, 2, 3, 1))

                    for j in range(self.batch_size):
                        save_unnormalized_image(
                            image = unnormalize(S_neg_intermediate_images[j,:,:,:]),  
                            path = S_neg_current_iteration_images_path[i * self.batch_size + j])

                    # Output information every 100 batches.
                    if i % 100 == 0:
                        print(("Sampler: Cascade {0}, iteration {1}, batch {2}, " + 
                            "time {3}, S_loss {4}").format(
                            cascade, iteration, i, time.time() - start_time, S_loss), file=logfile)

                # After current iteration, new negative images will be added into the set of
                # negative images. Note that we keep all previous pseudo-negative images
                # to prevent the classifier forgetting what it has learned in previous stages
                
                neg_all_images_path += S_neg_current_iteration_images_path

                # Save last batch images in sampling pseudo-negatives stage.
                S_neg_intermediate_image_path = os.path.join(intermediate_image_root,
                    'S_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
                # In discriminator we save D_batch_images, but here we use 
                # S_intermediate_images. It is because we always use *_batch_images
                # to represent the images we put in the discriminator or sampler.
                # So G_neg_batch_images should be the "initial" images in current 
                # iteration and S_neg_intermediate_images is the generated images.
                save_unnormalized_images(images = unnormalize(S_neg_intermediate_images), 
                                        size = (sqrt_batch_size, sqrt_batch_size), 
                                        path = S_neg_intermediate_image_path)

                # Last cascade's generated negative images. More specifically, we only use
                # those images generated by last iteration of last cascade.
                S_neg_last_cascade_images_path = copy.deepcopy(S_neg_current_iteration_images_path)
                
                # Save the model.
                torch.save(self.discriminator.state_dict(), os.path.join(self.opts.ckpt_dir, 'cascade-{}.model').format(cascade))
                    
        logfile.close()