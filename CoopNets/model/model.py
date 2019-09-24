import os 
import time

import torchvision
import torchvision.transforms as transform
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

from utils.data_io import *

from tensorboardX import SummaryWriter

#from opts import *

###########################################################################Descriptor
class Descriptor(nn.Module):
    def __init__(self, opts):
        super(Descriptor, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, \
        #padding=0, dilation=1, groups=1, bias=True)
        #add zeros both side padding
        
        #input (N, in_channels, H_in, W_in)
        #output (N, in_channels, H_out, W_out)
        #H_out = (H_in + 2*padding[0] - dilation[0](kernel_size[0]-1) -1)/stride[0]
        #W_out = (W_in + 2*padding[1] - dilation[1](kernel_size[1]-1) -1)/stride[1]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(16*16*256, opts.z_size)
        #z_size = size of latent variables
        
        self.leakyrelu = nn.LeakyReLU()
        #initial parameters
        
    def forward(self, x):
        self.x = x
        #Why no dropout and batchnorm?
        conv1 = self.leakyrelu(self.conv1(x))
        conv2 = self.leakyrelu(self.conv2(conv1))
        conv3 = self.leakyrelu(self.conv3(conv2))
        #flatten, size(0) = batch_size
        conv3 = conv3.view(conv3.size(0), -1)
        out = self.fc(conv3)
        return out       


##############################################################################Generator
class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
        #padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        #add zeros both side kernel_size - 1 - padding
        #output_padding controls the additional size added to one side of the output shape
        
        #input (N, in_channels, H_in, W_in)
        #output (N, in_channels, H_out, W_out)
        #H_out = (H_in −1)×stride[0]−2×padding[0]+kernel_size[0]+output_padding[0]
        #W_out = (W_in −1)×stride[1]−2×padding[1]+kernel_size[1]+output_padding[1]
        
        #why 512 not 256
        self.deconv1 = nn.ConvTranspose2d(in_channels=opts.z_size, out_channels=512, kernel_size=4, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
       
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        self.z = z
        deconv1 = self.deconv1(z)
        deconv1 = self.bn1(deconv1)
        deconv1 = self.leakyrelu(deconv1)
        
        deconv2 = self.deconv2(deconv1)
        deconv2 = self.bn2(deconv2)
        deconv2 = self.leakyrelu(deconv2)
        
        deconv3 = self.deconv3(deconv2)
        deconv3 = self.bn3(deconv3)
        deconv3 = self.leakyrelu(deconv3)
        
        deconv4 = self.deconv4(deconv3)
        deconv4 = self.bn4(deconv4)
        deconv4 = self.leakyrelu(deconv4)
        
        deconv5 = self.deconv5(deconv4)
        out = self.tanh(deconv5)
        
        return out



#########################################################################################Descriptor_cifar
class Descriptor_cifar(nn.Module):
    def __init__(self, opts):
        super(Descriptor_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(8*8*256, opts.z_size)
        #z_size = size of latent variables
        
        self.leakyrelu = nn.LeakyRelu()
        #initial parameters
    
    def forward(self, x):
        conv1 = self.leakyrelu(self.conv1(x))
        conv2 = self.leakyrelu(self.conv2(conv1))
        conv3 = self.leakyrelu(self.conv3(conv2))
        
        conv3 = conv3.view(conv3.size(0), -1)
        out = self.fc(conv3)
        
        return out


###########################################################################################Generator_cifar
class Generator_cifar(nn.Module):
    def __init__(self, opts):
        super(Generator_cifar, self).__init__()
        self.deconv1 = nn.TransposeConv2d(in_channels=opts.z_size, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.deconv2 = nn.TransposeConv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.TransposeConv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.TransposeConv2d(in_channels=64, out_channels=3, kernels_size=5, stride=2, padding=2, output_padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.leakyrelu = nn.LeakyRelu()
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        self.z = z
        deconv1 = self.deconv1(z)
        deconv1 = self.bn1(deconv1)
        deconv1 = self.leakyrelu(deconv1)
        
        deconv2 = self.deconv2(deconv1)
        deconv2 = self.bn2(deconv2)
        deconv2 = self.leakyrelu(deconv2)
        
        deconv3 = self.deconv3(deconv2)
        deconv3 = self.bn2(deconv3)
        deconv3 = self.leakyrelu(deconv3)
        
        deconv4 = self.deconv4(deconv3)
        out = self.tanh(deconv4)
        
        return out

    
################################################################################################CoopNets
class CoopNets(nn.Module):
    def __init__(self, opts):
        super(CoopNets, self).__init__()
        self.img_size = opts.img_size
        self.num_chain = opts.nRow*opts.nCol #each pixel
        self.opts = opts
        
        if(opts.with_noise):
            print("Langevin Dynamics with noise")
        else:
            print("Langevin Dynamics without noise")
        
        if(opts.set=='cifar'):
            opts.img_size = 32
            print("training on cifar with image size: %i" %(self.img_size))
            
    
    #############################################langevin dynamics for generator
    #gradient of generator parameter = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{\sigma^{2}}(Y_{i}-g(X_{i};\alpha))
    #\frac{\partial g(X_{i};\alpha)}{\partial \alpha}
    
    #X_{\tau+1} = X_{\tau} + \frac{step_size^{2}}{2}\frac{\partial log(q(X,Y;\alpha))}{\partial X} 
    #+ step_size*U_{\tau}
    
    #log(q(X,Y;\alpha)) = \frac{1}{2\sigma^{2}}|Y-g(X_{\tau};\alpha)^{2}
    #+\frac{1}{2}|X_{\tau}^{2}|+constant
    
    #\frac{\partial log(q(X,Y;\alpha))}{\partial X} = -\frac{1}{\sigma^{2}}(Y_{i}-g(X_{i};\alpha)) - |X|
    # (Y_{i}-g(X_{i};\alpha)) is z.grad of MSE Loss, z is |X|

    #noise is U_{\tau}


    def langevin_dynamics_generator(self, z, obs, mode='train'):

        #tensor.detach() creates a tensor that shares storage with tensor that does not require grad. 
        #obs means observed data, Y
        obs = obs.detach()
        criterian = nn.MSELoss(size_average=False, reduce=True, reduction='mean')
        #The division by n can be avoided if one sets size_average to False.
        #To get a batch of losses, a loss per batch element, set reduce to False. 
        
        #run langevin_step_num_gen steps langevin dynamics
        for i in range(self.opts.langevin_step_num_gen):

            #torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
            #sizes (int...) – a sequence of integers defining the shape of the output tensor. 
            #out (Tensor, optional) – the output tensor
            noise = Variable(torch.randn(self.num_chain, self.opts.z_size, 1, 1).cuda()) 
            #shape self.num_chain * self.opts.z_size * 1 * 1
            
            z = Variable(z, requires_grad=True) 
            # could return gradients

            gen_sample = self.generator(z) 
            #batch of X_{i}, and obs is batch of Y_{i}
            
            
            gen_loss = 1.0 / (2.0 * self.opts.sigma_gen * self.opts.sigma_gen) * criterian(gen_sample, obs)
            #use MSE Loss to increase likelihood in langevin dynamics

            gen_loss.backward()

            grad = z.grad 
            # to update z, langevin dynamics one step
            
            z = z - 0.5 * self.opts.langevin_step_size_gen * self.opts.langevin_step_size_gen * (z + grad)

            #+ step_size*U_{\tau}
            if((self.opts.with_noise==True)and(mode=='train')):
                z += self.opts.langevin_step_size_gen * noise

        return z

    
    #########################################langevin dynamics for descriptor
    #gradient for generator \frac{1}{n}\sum_{i=1}^{n}\frac{\partial f(Y_{i};\theta)}{\partial \theta} 
    # - \frac{1}{\tilde{n}}\frac{\partial f(\tilde{Y}_{i};\theta)}{\partial \theta} 

    #Y_{\tau+1} = Y_{\tau} + \frac{step_size^{2}}{2}\frac{\partial f(Y_{\tau;\theta})}{\partial Y} 
    # - \frac{step_size^{2}}{2}\frac{Y_{\tau}} + step_size*U_{\tau}

    ##grad = \frac{\partial f(Y_{\tau;\theta})}{\partial Y_{\tau}}, use autograd, do backward for all element of f(Y_{\tau;\theta})

    def langevin_dynamics_descriptor(self, x, mode='train'):
        
        #run langevin_step_num_gen steps langevin dynamics
        for i in range(self.opts.langevin_step_num_des):
            
            #dimension of x is 3
            noise = Variable(torch.randn(self.num_chain, 3, self.opts.img_size, self.opts.img_size).cuda())
            
            # clone it and turn x into a leaf variable so the grad won't be thrown away
            x = Variable(x.data, requires_grad=True)
            #"However, .data can be unsafe in some cases. 
            #Any changes on x.data wouldn’t be tracked by autograd, 
            #and the computed gradients would be incorrect if x is needed in a backward pass. 
            #A safer alternative is to use x.detach(), 
            #which also returns a Tensor that shares data with requires_grad=False, 
            #but will have its in-place changes reported by autograd if x is needed in backward."
            
            
            x_feature = self.descriptor(x)
            #x_feature is f(Y;\theta), x is Y
            
            #torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, 
            #create_graph=False, grad_variables=None)
            
            # do backward for all element of x_feature
            x_feature.backward(torch.ones(self.num_chain, self.opts.z_size).cuda())
            
            grad = x.grad
            
            # print ('x is : '+str(x[0]))
            # print ('x_grad is : '+str(grad[0]))
            
            x = x - 0.5 * self.opts.langevin_step_size_des * self.opts.langevin_step_size_des * \
                    (x / (self.opts.sigma_des * self.opts.sigma_des) - grad)
            
            #+ step_size*U_{\tau}
            if((self.opts.with_noise)and(mode=='train')):
                x += self.opts.langevin_step_size_des * noise
                
        return x
    
    
    
    def train(self):

        #load model
        if((self.opts.ckpt_des!=None)and(self.opts.ckpt_des!='None')):

            self.descriptor = torch.load(self.opts.ckpt_des).train()
            print('Loading Descriptor from ' + self.opts.ckpt_des + '...')

        else:

            if((self.opts.set=='scene')or(self.opts.set=='lsun')):

                self.descriptor = Descriptor(self.opts).cuda().train()
                print('Loading Descriptor without initialization...')

            elif(self.opts.set=='cifar'):

                self.descriptor = Descriptor_cifar(self.opts).cuda().train()
                print('Loading Descriptor_cifar without initialization...')

            else:

                raise NotImplementedError('The set should be either scene, lsun, or cifar')

        if((self.opts.ckpt_gen!=None)and(self.opts.ckpt_gen!='None')):

            self.generator = torch.load(self.opts.ckpt_gen).train()
            print('Loading Generator from ' + self.opts.ckpt_gen + '...')

        else:

            if((self.opts.set=='scene') or (self.opts.set=='lsun')):

                self.generator = Generator(self.opts).cuda().train()
                print('Loading Generator without initialization...')

            elif(self.opts.set=='cifar'):

                self.generator = Generator_cifar(self.opts).cuda().train()
                print('Loading Generator_cifar without initialization...')

            else:

                raise NotImplementedError('The set should be either scene, lsun or cifar')


        #load dataset
        batch_size = self.opts.batch_size   
        if((self.opts.set=='scene')or(self.opts.set=='cifar')):

            train_data = DataSet(os.path.join(self.opts.data_path, self.opts.category), \
                image_size=self.opts.img_size)

        else:

            train_data = torchvision.datasets.LSUN(root=self.opts.data_path, classes=['bedroom_train'],\
                                                    transform=transform.Compose([transform.Resize(self.img_size), \
                                                        transform.ToTensor(), ]))
            #root (string) – Root directory for the database files.
            #classes (string or list) – One of {‘train’, ‘val’, ‘test’} or a list of categories to load. e,g. [‘bedroom_train’, ‘church_train’].
            #transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
            #target_transform (callable, optional) – A function/transform that takes in the target and transforms it.

            #transform.Compose([transform.Resize(self.img_size), transform.ToTensor(), ], do resize transform             

        num_batch = int(np.ceil(len(train_data)/batch_size))

        #sample_results = np.random.randn(self.num_chain*num_batch, self.img_size, self.img_size, 3)

        des_optimizer = torch.optim.Adam(self.descriptor.parameters(), lr=self.opts.lr_des, \
                                        beta=[self.opts.beta1_des, 0.999])

        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opts.lr_gen, \
                                        beta=[self.opts.beta1_gen, 0.999])

        #params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        #lr (float, optional): learning rate (default: 1e-3)
        #betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        #eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        #weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        #amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_(default: False)
        
        
        #ckpt dir and output dir
        if(not os.path.exists(self.opts.ckpt_dir)):

            os.makedirs(self.opts.ckpt_dir)

        if(not os.path.exists(self.opts.output_dir)):

            os.makedirs(self.opts.output_dir)

        #log file
        log_file = open(self.opts.ckpt_dir + '/log', 'a+')

        #Loss
        mse_loss = nn.MSELoss(size_average=False, reduce=True, reduction='mean')

        #tensorboardX writer
        writer = SummaryWriter()

        #training
        for epoch in range(self.opts.num_epoch):

            start_time = time.time()
            gen_loss_list, des_loss_list, recon_loss_list = [], [], []

            for i in range(num_batch):
                
                obs_data = train_data[i*batch_size : min((i+1)*batch_size, len(train_data))]
                obs_data = Variable(torch.Tensor(obs_data).cuda())

                #G0 
                #Generator generates samples
                #shape NCHW
                z = torch.randn(self.num_chain, self.opts.z_size, 1, 1)
                z = Variable(z.cuda(), requires_grad=True)

                gen_samples = self.generator(z)

                #D1
                #Descriptor Langevin Dynamics
                if(self.opts.langevin_step_num_des>0):
                    revised_samples = self.langevin_dynamics_descriptor(gen_samples)
                
                #G1
                #Generator Langevin Dynamics
                if(self.opts.langevin_step_num_des>0):
                    z = self.langevin_dynamics_generator(z, revised_samples)

                #D2
                #Descriptor update parameters
                obs_feature = self.descriptor(obs_data)
                revised_feature = self.descriptor(revised_samples)

                des_loss = (torch.mean(obs_feature, 0) - torch.mean(revised_feature, 0)).sum()

                des_optimizer.zero_grad()
                des_loss.backward()
                des_optimizer.step()

                #G2
                #Generator update parameters
                #supervised learning, data-gen_samples, label-revised_samples
                init_gen_samples = gen_samples.detach() #use for reconstruction loss

                if self.opts.langevin_step_num_gen > 0:
                    gen_samples = self.generator(z)
                
                gen_samples = gen_samples.detach() # won't change
                
                #gen_loss = 0.5 * self.opts.sigma_gen * self.opts.sigma_gen * mse_loss(gen_samples, revised_samples.detach())
                #gen_loss = mse_loss(gen_samples, revised_samples.detach())
                gen_loss = 1.0 / (2 * self.opts.sigma_gen * self.opts.sigma_gen) * mse_loss(gen_samples, revised_samples.detach())

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()


                # Compute reconstruction loss
                recon_loss = mse_loss(revised_samples, init_gen_samples)

                gen_loss_list.append(gen_loss.cpu().data)
                des_loss_list.append(des_loss.cpu().data)
                recon_loss_list.append(recon_loss.cpu().data)

                writer.add_scalar('data/gen_loss', dummy_s1[0], n_iter)
                writer.add_scalar('data/des_loss', dummy_s2[0], n_iter)
                writer.add_scalar('data/recon_loss', dummy_s2[0], n_iter)

                # TO-FIX (confliction between pytorch and tf)
                # if opts.incep_interval>0, compute inception score each [incep_interval] epochs.
                # if self.opts.incep_interval > 0:
                #     import inception_model
                #     if epoch % self.opts.incep_interval == 0:
                #         inception_log_file = os.path.join(self.opts.output_dir, 'inception.txt')
                #         inception_output_file = os.path.join(self.opts.output_dir, 'inception.mat')
                #         sample_results_partial = revised[:len(train_data)]
                #         sample_results_partial = np.minimum(1, np.maximum(-1, sample_results_partial))
                #         sample_results_partial = (sample_results_partial + 1) / 2 * 255
                #         # sample_results_list = sample_results.copy().swapaxes(1, 3)
                #         # sample_results_list = np.split(sample_results, len(sample_results), axis=0)
                #         m, s = get_inception_score(sample_results_partial)
                #         fo = open(inception_log_file, 'a')
                #         fo.write("Epoch {}: mean {}, sd {}".format(epoch, m, s))
                #         fo.close()
                #         inception_mean.append(m)
                #         inception_sd.append(s)
                #         sio.savemat(inception_output_file,
                #                     {'mean': np.asarray(inception_mean), 'sd': np.asarray(inception_sd)})

                # save observed samples
                try:
                    col_num = self.opts.nCol
                    saveSampleResults(obs_data.cpu().data[:col_num * col_num], "%s/observed.png" % (self.opts.output_dir),
                                    col_num=col_num)
                except:
                    print('Error when saving obs_data. Skip.')
                    continue

                # save revised samples (generated samples -> langevin dynamics)
                saveSampleResults(revised_samples.cpu().data, "%s/des_%03d.png" % (self.opts.output_dir, epoch + 1),
                                col_num=self.opts.nCol)
                
                # save generated samples
                saveSampleResults(gen_samples.cpu().data, "%s/gen_%03d.png" % (self.opts.output_dir, epoch + 1),
                                col_num=self.opts.nCol)

                end_time = time.time()
                print('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                    'time: {:.2f}s'.format(epoch + 1, self.opts.num_epoch, np.mean(des_loss_list),
                                            np.mean(gen_loss_list), np.mean(recon_loss_list),
                                            end_time - start_time))

                # python 3
                print('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                    'time: {:.2f}s'.format(epoch, self.opts.num_epoch, np.mean(des_loss_list), np.mean(gen_loss_list),
                                            np.mean(recon_loss_list),
                                            end_time - start_time), file=log_file)
                # python 2.7
                # print >> logfile, ('Epoch #{:d}/{:d}, des_loss: {:.4f}, gen_loss: {:.4f}, recon_loss: {:.4f}, '
                #     'time: {:.2f}s'.format(epoch,self.opts.num_epoch, np.mean(des_loss_epoch), np.mean(gen_loss_epoch), np.mean(recon_loss_epoch),
                #                            end_time - start_time))


                if epoch % self.opts.log_epoch == 0:
                    torch.save(self.descriptor, self.opts.ckpt_dir + '/des_ckpt_{}.pth'.format(epoch))
                    torch.save(self.generator, self.opts.ckpt_dir + '/gen_ckpt_{}.pth'.format(epoch))
            
            log_file.close()

    
    def test(self):
        assert self.opts.ckpt_gen is not None, 'Please specify the path to the checkpoint of generator.'
        assert self.opts.ckpt_des is not None, 'Please specify the path to the checkpoint of generator.'
        print('===Test on ' + self.opts.ckpt_gen + ' and ' + self.opts.ckpt_des+' ===')

        # load model
        generator = torch.load(self.opts.ckpt_gen).eval()
        descriptor = torch.load(self.opts.ckpt_des).eval()

        if not os.path.exists(self.opts.output_dir):
            os.makedirs(self.opts.output_dir)

        test_batch=int(np.ceil(self.opts.test_size/self.opts.nRow/self.opts.nCol))
        print('===Generated images saved to %s ===' % (self.opts.output_dir))

        for i in range(test_batch):

            z = torch.randn(self.num_chain, self.opts.z_size, 1, 1)
            z = Variable(z.cuda())
            #no require grad

            #G0
            #Generator generates samples
            gen_samples = generator(z)

            #D1
            #Descriptor Langevin Dynamics
            revised_samples = self.langevin_dynamics_descriptor(gen_samples, 'test')

            #for s in range(self.opts.langevin_step_num_des):
                # clone it and turn x into a leaf variable so the grad won't be thrown away
                #gen_res = Variable(gen_res.data, requires_grad=True)
                #gen_res_feature = descriptor(gen_res)
                #gen_res_feature.backward(torch.ones(self.num_chain, self.opts.z_size).cuda())
                #grad = gen_res.grad
                #gen_res = gen_res - 0.5 * self.opts.langevin_step_size_des * self.opts.langevin_step_size_des * \
                                    #(gen_res / self.opts.sigma_des / self.opts.sigma_des - grad)


            if self.opts.score:

                revised_samples=revised_samples.detach().cpu()

                for img_no,img in enumerate(revised_samples):

                    if i*self.num_chain+img_no+1>self.opts.test_size:
                        break

                    print('Generating {:05d}/{:05d}'.format(i*self.num_chain+img_no+1,self.opts.test_size))
                    saveSampleResults(img[None,:,:,:], "%s/testres_%03d.png" % (self.opts.output_dir,
                                                                                   i*self.num_chain+img_no+1 ),
                                      col_num=1,margin_syn=0)
            else:

                revised_samples = revised_samples.detach().cpu()
                print('Generating {:05d}/{:05d}'.format(i+1, test_batch))
                saveSampleResults(revised_samples, "%s/testres_%03d.png" % (self.opts.output_dir,
                                                                i+1),
                                      col_num=self.opts.nCol, margin_syn=0)

        print ('===Image generation done.===')







        

