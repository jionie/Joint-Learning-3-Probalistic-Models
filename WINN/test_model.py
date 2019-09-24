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
from torch.autograd import Variable, grad
from torch import autograd

import gc

from utils.data_io import *

from WINN_utils import *

from opts import *

DIM=64
OUTPUT_DIM=64*64*3
LAMBDA = 10

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

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = LayerNorm(input_dim)
            self.bn2 = LayerNorm(input_dim)
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = LayerNorm(output_dim)
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
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


class GoodGenerator(nn.Module):
    def __init__(self, dim=DIM,output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4*4*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, OUTPUT_DIM)
        return output


class GoodDiscriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim

        self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down')
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down')
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down')
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down')
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, 3, 64 ,64)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output


class INN_Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64, batch_norm=True):
        super(INN_Discriminator, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.batch_norm = batch_norm
        
        self.conv1 = MyConvo2d(self.in_channels, int(dim/2), 3, 1)
        self.swish = swish()
        
        self.conv2 = MyConvo2d(int(dim/2), dim, 3, 1)
        self.layernorm1 = LayerNorm(dim)
        
        self.meanpoolconv1 = MeanPoolConv(dim, dim, 3, 1)
        self.layernorm2 = LayerNorm(dim)
        
        self.conv3 = MyConvo2d(dim, dim*2, 3, 1)
        self.layernorm3 = LayerNorm(dim*2)
        
        self.meanpoolconv2 = MeanPoolConv(dim*2, dim*2, 3, 1)
        self.layernorm4 = LayerNorm(dim*2)
        
        self.conv4 = MyConvo2d(dim*2, dim*4, 3, 1)
        self.layernorm5 = LayerNorm(dim*4)
        
        self.meanpoolconv3 = MeanPoolConv(dim*4, dim*4, 3, 1)
        self.layernorm6 = LayerNorm(dim*4)
        
        self.conv5 = MyConvo2d(dim*4, dim*8, 3, 1)
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
        self.conv1 = MyConvo2d(8*self.dim, 4*self.dim, 5, 1)
        self.upsample1 = nn.Upsample(size=8, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm1 = LayerNorm(4*self.dim)
        
        self.conv2 = MyConvo2d(4*self.dim, 2*self.dim, 5, 1)
        self.upsample2 = nn.Upsample(size=16, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm2 = LayerNorm(2*self.dim)
        
        self.conv3 = MyConvo2d(2*self.dim, self.dim, 5, 1)
        self.upsample3 = nn.Upsample(size=32, scale_factor=None, mode='nearest', align_corners=None)
        self.layernorm3 = LayerNorm(self.dim)
        
        self.conv4 = MyConvo2d(self.dim, 3, 5, 1)
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
    def __init__(self, opts, discriminator_model=None, in_channels=3, dim=64, cascades=4, iterations_per_cascade=100, discriminator_train_steps=3, LAMBDA=10):
        super(WINN, self).__init__()
        self.batch_size = opts.batch_size
        self.in_channels = in_channels
        self.img_size = opts.img_size
        self.dim = dim
        self.num_chain = opts.nRow*opts.nCol #each image in final result
        self.opts = opts
        self.Noise_Provider = Noise(100, dim=64)
        self.cascades = cascades
        self.iterations_per_cascade = iterations_per_cascade
        self.discriminator_train_steps = discriminator_train_steps
        self.LAMBDA = LAMBDA

        if(discriminator_model!=None):
            self.discriminator = torch.load(discriminator_model).train()
            print('Loading Discriminator from ' + discriminator_model + '...')
        else:
            self.discriminator = INN_Discriminator().train()
            print('Loading Discriminator without initialization...')
        
        if(opts.with_noise):
            print("Langevin Dynamics with noise")
        else:
            print("Langevin Dynamics without noise")
        
        if(opts.set=='cifar'):
            opts.img_size = 32
            print("training on cifar with image size: %i" %(self.img_size))


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
        for i in range(self.opts.langevin_step_num_des):
            
            #dimension of x is 3
            noise = Variable(torch.randn(self.num_chain, 3, self.opts.img_size, self.opts.img_size).cuda())
            #"However, .data can be unsafe in some cases. 
            #Any changes on x.data wouldn’t be tracked by autograd, 
            #and the computed gradients would be idiscriminator_train_steporrect if x is needed in a backward pass. 
            #A safer alternative is to use x.detach(), 
            #which also returns a Tensor that shares data with requires_grad=False, 
            #but will have its in-place changes reported by autograd if x is needed in backward."
            
            # clone it and turn x into a leaf variable so the grad won't be thrown away
            x = Variable(x.data, requires_grad=True)
            
            #gradient is torch.ones(self.num_chain, self.opts.z_size).cuda()
            
            x_feature = self.discriminator(x)
            #x_feature is f(x;\theta) which is \ln(p(y=1|x,\theta)/p(y=0|x,\theta))
            
            #torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, 
            #create_graph=False, grad_variables=None)
            
            # do backward for all element of x_feature
            x_feature.backward(torch.ones(self.num_chain, self.opts.z_size).cuda())
            
            #grad = \frac{\partial f(x;\theta)}{\partial x}
            grad = x.grad
            
            # print ('x is : '+str(x[0]))
            # print ('x_grad is : '+str(grad[0]))
            
            x = x + 0.5 * self.opts.langevin_step_size_dis * self.opts.langevin_step_size_dis * grad
            
            #+ step_size*U_{\tau}
            if self.opts.with_noise:
                x += self.opts.langevin_step_size_dis * noise
                
        return x 
        
    def train(self, LAMBDA=10.0):
        
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

        self.Noise_Provider.eval().cuda()
        if not os.path.exists(neg_image_root):
            os.makedirs(neg_image_root)
            # Generate initial pseudo-negative images
            # In fact, the name of image has format 
            #     {cascade}_{next iteration}_{i}.png
            # where cascade means current cascade model, next iteration means
            # next iteration of sampler and discriminator training, and i means
            # the index of images.
            neg_image_path = os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png')
            neg_init_images_count = 10000
            neg_init_images_path = [neg_image_path.format(0, 0, i) for i in range(neg_init_images_count)]

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

        negatives_list = os.listdir(neg_image_root) # dir is your directory path
        image_shape = train_data.images.shape[1:]
        self.Noise_Provider.cpu()
        torch.cuda.empty_cache()
        ######################################################
        print("Positive images {0}, negative images {1}, image shape {2}"\
            .format(train_data.images.shape[0], len(negatives_list), image_shape), file=logfile)
        
        ######################################################################
        # Training stage 3: Cascades training.
        ######################################################################
        print("Training stage 2: Cascades training...", file=logfile)

        # Prepare for the initial images to feed the sampler. In fact, it is 
        # because we always use negative images in last cascade as the "initial"
        # images to feed sampler in all iterations of current cascade.
        S_neg_last_cascade_images_path = copy.deepcopy(negatives_list)
        half_batch_size = int(self.batch_size/2) ## half of positive images, half of negative images
        

        for cascade in range(self.cascades):
        ######################################################################
        # Training stage 3.1: Iterations training.
        ######################################################################
        # One iteration means one time of discriminator training and one time
        # of sampling pseudo-negatives. One iteration training may contain multiple
        # batches for discriminator training and sampling pseudo-negatives.
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


                D_neg_iteration_images_count = D_pos_iteration_images_count
                train_data_negatives = DataSet(neg_image_root,image_size=self.opts.img_size)
                train_data_negatives.shuffle()
                D_neg_iteration_images = train_data_negatives[:D_neg_iteration_images_count]

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
                self.discriminator.cuda()
                #summary(self.discriminator, (3, 64, 64))

                # Training for self.discriminator_train_steps steps
                for discriminator_train_step in range(self.discriminator_train_steps):
                    for i in range(D_iteration_count_of_batch):

                        one = torch.FloatTensor([1]).cuda()
                        mone = (one * -1)

                        # Load images for this batch in discriminator.
                        D_pos_batch_images = D_pos_iteration_images[i * half_batch_size : (i + 1) * half_batch_size, :, :, :]
                        D_neg_batch_images = D_neg_iteration_images[i * half_batch_size : (i + 1) * half_batch_size, :, :, :]
                        #print(D_pos_batch_images.shape)
                        # Normalize.
                        D_pos_batch_images = normalize(np.array(D_pos_batch_images)).astype(np.float32)
                        D_neg_batch_images = normalize(np.array(D_neg_batch_images)).astype(np.float32)

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

                        D_loss = D_neg_batch_logits - D_pos_batch_logits + gradient_penalty
                        Wasserstein_D = D_pos_batch_logits - D_neg_batch_logits


                        # update weightss
                        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.opts.lr_dis,
                                         betas=[self.opts.beta1_dis, 0.9])
                        dis_optimizer.zero_grad()
                        dis_optimizer.step()
                        
                        print("Discriminator Loss: ", D_loss.detach().cpu().numpy())
                        #del D_neg_batch_images_cuda, D_pos_batch_images_cuda
                        #torch.cuda.empty_cache()
                        
                        

                       

        logfile.close()


def main():
    opt=opts().parse()
    model=WINN(opt)
    #cuda_available = torch.cuda.is_available()
    #device = torch.device("cuda" if cuda_available else "cpu")
    #model.to(device)
   
    if opt.test:
        model.test()
    else:
        model.train()

if __name__=='__main__':
    main()
        


