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
from utils.save_image import *
from inception_score import *
from inception_model import *
from frechet_inception_distance import *
from opts import *

from opts import *

""" from metric.inception_score import *
from metric.frechet_inception_distance import * """

import random

import cv2
import os
import math
import numpy as np
import cv2
import scipy.io as sio
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


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

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64, batch_norm=True):
        super(Discriminator, self).__init__()
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


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt

        ### (H-1)*stride + kernel_size - 2*padding + output_padding
        self.fc = nn.Linear(128, 2 * 2 * 4 * self.opt.z_size)
        self.bn0 = nn.BatchNorm1d(2 * 2 * 4 * self.opt.z_size)
        self.convt1 = nn.ConvTranspose2d(4 * self.opt.z_size, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt5 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU()
        self.swish = swish()
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()

    def forward(self, z):
        
        z = z.view(-1, self.opt.z_size)
        z = self.fc(z)
        #z = self.bn0(z)
        #z = z.view(-1, 2 * 2 * 4 * self.opt.z_size)
        #z = self.leakyrelu(z)
        z = z.view(-1, 4 * self.opt.z_size, 2, 2)

        out = self.convt1(z)
        out = self.bn1(out)
        out = self.swish(out)
        out = self.convt2(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.convt3(out)
        out = self.bn3(out)
        out = self.swish(out)
        out = self.convt4(out)
        out = self.bn4(out)
        out = self.swish(out)
        out = self.convt5(out)
        out = self.tanh(out)
        return out


class Generator_cifar(nn.Module):
    def __init__(self, opt):
        super(Generator_cifar, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(128, 2 * 2 * 4 * self.opt.z_size)
        self.bn0 = nn.BatchNorm1d(2 * 2 * 4 * self.opt.z_size)
        self.convt1 = nn.ConvTranspose2d(4 * self.opt.z_size, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU()
        self.swish = swish()
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()

    def forward(self, z):

        z = z.view(-1, self.opt.z_size)
        z = self.fc(z)
        #z = self.bn0(z)
        #z = z.view(-1, 2 * 2 * 4 * self.opt.z_size)
        #z = self.leakyrelu(z)
        z = z.view(-1, 4 * self.opt.z_size, 2, 2)

        out = self.convt1(z)
        out = self.bn1(out)
        out = self.swish(out)
        out = self.convt2(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.convt3(out)
        out = self.bn3(out)
        out = self.swish(out)
        out = self.convt4(out)
        out = self.tanh(out)
        return out



class Joint(nn.Module):
    def __init__(self, opts, discriminator_model=None, generator_model=None, \
                 in_channels=3, iterations=400, k=3, LAMBDA=10):
        
        super(Joint, self).__init__()
        self.batch_size = opts.batch_size
        self.in_channels = in_channels

        if(opts.set=='cifar'):
            opts.img_size = 32
            self.dim = 32
            print("training on cifar with image size: %i" %(opts.img_size))
            
            if(discriminator_model!=None):
                self.discriminator = Discriminator()
                self.discriminator.load_state_dict(torch.load(discriminator_model), strict=False)
                self.discriminator.cuda()
                print('Loading Discriminator from ' + discriminator_model + '...')
            else:
                self.discriminator = Discriminator().apply(init_weights).cuda()
                print('Initializing Discriminator without initialization...') 
                
            if(generator_model!=None):
                self.generator = Generator_cifar(opts)
                self.generator.load_state_dict(torch.load(generator_model), strict=False)
                self.generator.cuda()
                print('Loading Generator_cifar from ' + generator_model + '...')
            else:
                self.generator = Generator_cifar(opts).apply(init_weights)
                print('Initializing Generator_cifar without initialization...') 
                
        elif((opts.set == 'scene') or (opts.set == 'lsun') or (opts.set == 'svhn')):
            opts.img_size = 64
            self.dim = 64
            if(discriminator_model!=None):
                self.discriminator = Discriminator()
                self.discriminator.load_state_dict(torch.load(discriminator_model), strict=False)
                self.discriminator.cuda()
                print('Loading Discriminator from ' + discriminator_model + '...')
            else:
                self.discriminator = Discriminator().apply(init_weights).cuda()
                print('Initializing Discriminator without initialization...') 
                
            if(generator_model!=None):
                self.generator = Generator(opts)
                self.generator.load_state_dict(torch.load(generator_model), strict=False)
                self.generator.cuda()
                print('Loading Generator from ' + generator_model + '...')
            else:
                self.generator = Generator(opts).apply(init_weights).cuda()
                print('Initializing Generator without initialization...') 
                
        else:
            raise NotImplementedError('The set should be either scene, lsun, svhn, or cifar')
            
        

        self.img_size = opts.img_size
        self.opts = opts
        self.iterations = iterations
        self.k = k
        self.LAMBDA = LAMBDA
        self.num_chain = opts.nRow * opts.nCol
        
        
    def calc_gradient_penalty(self, real_data, fake_data):

        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(self.batch_size, int(real_data.nelement()/self.batch_size)).contiguous().view(self.batch_size, \
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
    
    def train(self):
        
        start_time = time.time()
        
        if not os.path.exists(self.opts.ckpt_dir):
            os.makedirs(self.opts.ckpt_dir)
        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)

        logfile = open(self.opts.ckpt_dir + '/log', 'w+')
        
        # Prepare for root directory of intermediate image.
        intermediate_image_root = os.path.join(self.opts.output_dir, "intermediate")

        if not os.path.exists(intermediate_image_root):
            os.makedirs(intermediate_image_root)
            
        ######################################################################
        # Training stage 1: Load positive images.
        ######################################################################
        print("Training stage 1: Load positive images...", file=logfile)

        num_train = 1000

        #categories for svhn are 'train', 'test', 'extra'
        if self.opts.set == 'scene' or self.opts.set == 'cifar' or self.opts.set == 'svhn':
            train_data = DataSet(os.path.join(self.opts.data_path, self.opts.category), 
                                 image_size=self.img_size)[:num_train]
        else:
            train_data = torchvision.datasets.LSUN(root=self.opts.data_path,
                                                   classes=['bedroom_train'],
                                                   transform=transforms.Compose([transforms.Resize(self.img_size),
                                                   transforms.ToTensor(), ]),
                                                   )

        print("Training with ", num_train, " samples")
   
        ######################################################################
        # Training stage 2: Training.
        ######################################################################
        print("Training stage 2: Training...", file=logfile)

        #half_batch_size = int(self.batch_size/2) ## half of positive images, half of negative images
        
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=self.opts.lr_dis,\
                                        betas=[self.opts.beta1_dis, 0.9])
        
        for iteration in range(self.iterations):

            """ if(iteration+1%100==0):
                self.k-=1 """

            print("Iteration: ", iteration)

            num_workers = 12
            D_pos_iteration_images_tensor = torch.tensor(np.array(train_data), dtype=torch.float)[:num_train]
            D_pos_dataset = torch.utils.data.TensorDataset(D_pos_iteration_images_tensor)
            D_pos_loader = torch.utils.data.DataLoader(D_pos_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

            num_batches = int(math.ceil(len(train_data) / self.batch_size))
            
            min_D_batch_pos_loss = np.inf
            max_D_batch_pos_loss = -np.inf

            self.discriminator.train().cuda()
            self.generator.train().cuda()
            
            # Training for self.k steps
            for i, (D_pos_data,) in enumerate(D_pos_loader):
            #for i in range(D_iteration_count_of_batch):

                print("The " + str(i+1) + "th batch of " + str(num_batches) + " batches", file=logfile)
                print("The " + str(i+1) + "th batch of " + str(num_batches) + " batches")
                
                ######################################################################
                # Training stage 2.1: Train the discriminator.
                ######################################################################
                for discriminator_train_step in range(self.k):

                    ##### Generate pseudo negatives
                    z = torch.randn(self.batch_size, self.opts.z_size, 1, 1).cuda()
                    D_neg_batch_images_cuda = self.generator(z)
                    
                    one = torch.FloatTensor([1]).cuda()
                    mone = (one * -1)

                    dis_optimizer.zero_grad()

                    #D_pos_batch_images = D_pos_iteration_images[i * half_batch_size : (i + 1) * half_batch_size, :, :, :]
                    #D_pos_batch_images_cuda = torch.tensor(D_pos_batch_images, dtype=torch.float).cuda()

                    D_pos_batch_images_cuda = D_pos_data.cuda()

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
                    print("Discriminator Loss: ", D_loss.detach().cpu().numpy(), "Wasserstein Distance: ", Wasserstein_D.detach().cpu().numpy(), file=logfile)
                    #torch.cuda.empty_cache()

                    if (discriminator_train_step == self.k - 1):
                        # Positive samples' loss after training in current iteration.
                        # It will be used as an early stopping threshold when we generate pseudo-negative samples
                        D_batch_pos_loss = D_pos_batch_logits.detach().cpu().numpy()
                        if (D_batch_pos_loss < min_D_batch_pos_loss):
                            min_D_batch_pos_loss = D_batch_pos_loss
                        if (D_batch_pos_loss > max_D_batch_pos_loss):
                            max_D_batch_pos_loss = D_batch_pos_loss

                if(i==0):
                    observed_images = D_pos_data.detach().cpu().numpy()
                else:
                    observed_images = np.concatenate((observed_images, D_pos_data.detach().cpu().numpy()), axis=0)
                
                """ if(i==num_batches-1):
                    # Save last col_num * col_num real images in discriminator training.
                    col_num = self.opts.nCol
                    observe_path = os.path.join(intermediate_image_root, 'real_{0}.png').format(iteration)
                    saveSampleResults(observed_images[observed_images.shape[0]-col_num*col_num:], observe_path, col_num=col_num) """
                   

                print("Discriminator: iteration {0}, Critic {1}, time {2}, D_loss {3}, D_pos_loss {4}, {5}"\
                    .format(iteration, discriminator_train_step, time.time() - start_time, D_loss.detach().cpu().numpy(), \
                    min_D_batch_pos_loss, max_D_batch_pos_loss), file=logfile) 

            
                thres_ = np.random.uniform(min_D_batch_pos_loss, max_D_batch_pos_loss)

                print("maximum generator num steps: ", self.opts.step_num_gen)

                ######################################################################
                # Training stage 2.2: Synthetic langevin dynamics
                ######################################################################
                z = torch.randn(self.batch_size, self.opts.z_size, 1, 1).cuda()
                D_neg_batch_images_cuda = self.generator(z)
                gen_res = Variable(D_neg_batch_images_cuda, requires_grad=True)
                S_optimizer = torch.optim.Adam([gen_res], lr=self.opts.lr_s, betas=[self.opts.beta1_s, 0.9])

                for j in range(self.opts.step_num_gen):
                    # Optimize.
                    S_optimizer.zero_grad()

                    S_gen_batch_logits = self.discriminator(gen_res)
                    S_loss = S_gen_batch_logits.mean()
                    S_loss.backward(-1*torch.FloatTensor([1]).cuda())

                    S_optimizer.step()

                    if(j%10==0):
                        S_loss = S_loss.detach().cpu().numpy()
                        if S_loss >= thres_:
                            break

                    if(j%100==0):
                        print("step: ", j, "S_loss: ", S_loss, "min_pos_loss: ", min_D_batch_pos_loss, "max_pos_loss: ", max_D_batch_pos_loss, \
                            "thres: ", thres_)
                        print("step: ", j, "S_loss: ", S_loss, "min_pos_loss: ", min_D_batch_pos_loss, "max_pos_loss: ", max_D_batch_pos_loss, \
                            "thres: ", thres_, file=logfile)

                ######################################################################
                # Training stage 2.3: Training Generator
                ######################################################################
                mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
                gen_loss = np.inf
                #synthetic_image = torch.tensor(synthetic_image, dtype=torch.float)
                gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=self.opts.lr_gen, betas=[self.opts.beta1_gen, 0.9])
                gen_iter = 0
                while(True):
                    gen_optimizer.zero_grad()
                    
                    gen_res_ori = self.generator(z)
                    gen_loss = 0.5 * self.opts.sigma_gen * self.opts.sigma_gen * mse_loss(gen_res_ori, gen_res)
                    gen_loss.backward()
                    gen_optimizer.step()

                    if(gen_iter%10==0):
                        gen_loss = gen_loss.detach().cpu().numpy()
                        if(gen_loss<1e-1):
                            break

                    if(gen_iter%100==0):
                        print("Generator optimization iteration: ", gen_iter, "Generator loss: ", gen_loss)
                        print("Generator optimization iteration: ", gen_iter, "Generator loss: ", gen_loss, file=logfile)

                    gen_iter += 1

                if(i==0):
                    des_images = gen_res.detach().cpu().numpy()
                    gen_images = self.generator(z).detach().cpu().numpy()

                else:
                    des_images = np.concatenate((des_images, gen_res.detach().cpu().numpy()), axis=0)
                    gen_images = np.concatenate((gen_images, self.generator(z).detach().cpu().numpy()), axis=0)
                

            col_num = self.opts.nCol
    
            # save result every 100 iterations
            if(iteration%100==0):
            
                torch.save(self.discriminator.state_dict(), os.path.join(self.opts.ckpt_dir, 'discriminator_'+str(iteration)))
                torch.save(self.generator.state_dict(), os.path.join(self.opts.ckpt_dir, 'generator_'+str(iteration)))

                observe_path = os.path.join(intermediate_image_root, 'Real_iteration_{0}.png').format(iteration)
                S_gen_intermediate_image_path = os.path.join(intermediate_image_root, 'Gen_iteration_{0}.png').format(iteration)
                S_des_intermediate_image_path = os.path.join(intermediate_image_root, 'Des_iteration_{0}.png').format(iteration)

                saveSampleResults(observed_images[observed_images.shape[0]-col_num*col_num:], observe_path, col_num=col_num)
                saveSampleResults(gen_images[gen_images.shape[0]-col_num*col_num:], S_gen_intermediate_image_path, col_num=col_num)
                saveSampleResults(des_images[des_images.shape[0]-col_num*col_num:], S_des_intermediate_image_path, col_num=col_num)
                
        logfile.close()
    
    def test(self):
        generate_image_root = os.path.join(self.opts.output_dir, "generate")
        des_image_root = os.path.join(self.opts.output_dir, "des")

        if not os.path.exists(generate_image_root):
            os.makedirs(generate_image_root)

        if not os.path.exists(des_image_root):
            os.makedirs(des_image_root)


        self.discriminator.eval().cuda()
        self.generator.eval().cuda()

        num_test = 1000

        if self.opts.set == 'scene' or self.opts.set == 'cifar' or self.opts.set == 'svhn':
            train_data = DataSet(os.path.join(self.opts.data_path, self.opts.category), 
                                 image_size=self.img_size)[:num_test]
        else:
            train_data = torchvision.datasets.LSUN(root=self.opts.data_path,
                                                   classes=['bedroom_train'],
                                                   transform=transforms.Compose([transforms.Resize(self.img_size),
                                                   transforms.ToTensor(), ]))[:num_test]

        ######generate sample size dataset as the real dataset
        num_workers = 12
        D_pos_iteration_images_tensor = torch.tensor(np.array(train_data), dtype=torch.float)
        D_pos_dataset = torch.utils.data.TensorDataset(D_pos_iteration_images_tensor)
        D_pos_loader = torch.utils.data.DataLoader(D_pos_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))

        for i, (D_pos_data,) in enumerate(D_pos_loader):

            print("The " + str(i+1) + "th batch of " + str(num_batches) + " batches")

            ##### Generate pseudo negatives
            z = torch.randn(self.batch_size, self.opts.z_size, 1, 1).cuda()
            D_neg_batch_images_cuda = self.generator(z)
            D_pos_batch_images_cuda = D_pos_data.cuda()

            #wgan gp
            
            D_pos_batch_images_cuda_v = autograd.Variable(D_pos_batch_images_cuda)
            D_pos_batch_logits = self.discriminator(D_pos_batch_images_cuda_v)
            D_pos_batch_logits = D_pos_batch_logits.mean()

            D_batch_pos_loss = D_pos_batch_logits.detach().cpu().numpy()

            min_D_batch_pos_loss = np.inf
            max_D_batch_pos_loss = -np.inf

            if (D_batch_pos_loss < min_D_batch_pos_loss):
                min_D_batch_pos_loss = D_batch_pos_loss
            if (D_batch_pos_loss > max_D_batch_pos_loss):
                max_D_batch_pos_loss = D_batch_pos_loss

            thres_ = np.random.uniform(min_D_batch_pos_loss, max_D_batch_pos_loss)
            
            z = torch.randn(self.batch_size, self.opts.z_size, 1, 1).cuda()
            D_neg_batch_images_cuda = self.generator(z)
            gen_res = Variable(D_neg_batch_images_cuda, requires_grad=True)
            S_optimizer = torch.optim.Adam([gen_res], lr=self.opts.lr_s, betas=[self.opts.beta1_s, 0.9])

            for j in range(self.opts.step_num_gen):
                # Optimize.
                S_optimizer.zero_grad()

                S_gen_batch_logits = self.discriminator(gen_res)
                S_loss = S_gen_batch_logits.mean()
                S_loss.backward(-1*torch.FloatTensor([1]).cuda())

                S_optimizer.step()

                if(j%10==0):
                    S_loss = S_loss.detach().cpu().numpy()
                    if S_loss >= thres_:
                        break

                if(j%100==0):
                    print("step: ", j, "S_loss: ", S_loss, "min_pos_loss: ", min_D_batch_pos_loss, "max_pos_loss: ", max_D_batch_pos_loss, \
                        "thres: ", thres_)

           
            gen_images = self.generator(z).detach().cpu().numpy()
            des_images = gen_res.detach().cpu().numpy()
            
            for k in range(D_pos_data.shape[0]):
                gen_image_path = os.path.join(generate_image_root, 'Gen_batch_{0}_num_{1}.png').format(i, k)
                des_image_path = os.path.join(des_image_root, 'Des_batch_{0}_num_{1}.png').format(i, k)

                saveSampleResults(gen_images[k, :, :, :][np.newaxis, :], gen_image_path, col_num=1)
                saveSampleResults(des_images[k, :, :, :][np.newaxis, :], des_image_path, col_num=1)


    def get_score(self):
        if self.opts.set == 'scene' or self.opts.set == 'cifar':
            train_data = DataSet(os.path.join(self.opts.data_path, self.opts.category), 
                                    image_size=self.opts.img_size)
        else:
            train_data = torchvision.datasets.LSUN(root=self.opts.data_path,
                                        classes=['bedroom_train'],
                                        transform=transforms.Compose([transforms.Resize(self.opts.img_size),
                                        transforms.ToTensor(), ]))

        Real_image_tensor = torch.tensor(np.array(train_data), dtype=torch.float)

        generate_image_root = os.path.join(self.opts.output_dir, "generate")
        Generated_data = DataSet(generate_image_root, image_size=self.opts.img_size)
        Generated_image_tensor = torch.tensor(np.array(Generated_data), dtype=torch.float)


        Revised_image_root = os.path.join(self.opts.output_dir, "des")
        Revised_data = DataSet(Revised_image_root, image_size=self.opts.img_size)
        Revised_image_tensor = torch.tensor(np.array(Revised_data), dtype=torch.float)

        print("Inception score (mean, std) for real dataset: ", inception_score(Real_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))
        print("Inception score (mean, std) for genrated dataset: ", inception_score(Generated_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))
        print("Inception score (mean, std) for revised generated dataset: ", inception_score(Revised_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))


        Real_Generated_fid_value = calculate_fid_given_paths(Real_image_tensor, Generated_image_tensor, batch_size=50, cuda=True, dims=2048)
        Real_Revised_fid_value = calculate_fid_given_paths(Real_image_tensor, Revised_image_tensor, batch_size=50, cuda=True, dims=2048)

        print("Frechet Inception Distance between real dataset and generated dataset: ", Real_Generated_fid_value)
        print("Frechet Inception Distance between real dataset and revised generated dataset: ", Real_Revised_fid_value)


                
                    
