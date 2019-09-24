import os
import argparse

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-num_epoch', type=int, default=300, help='training epochs')
        self.parser.add_argument('-batch_size', type=int, default=100, help='training batch size')
        self.parser.add_argument('-img_size', type=int, default=64, help='output image size')
        self.parser.add_argument('-nRow', type=int, default=30, help='how many rows of images in the output')
        self.parser.add_argument('-nCol', type=int, default=30, help='how many columns of images in the output')
        self.parser.add_argument('-z_size', type=int, default=128, help='dimension of latent variable sample from latent space')
        self.parser.add_argument('-category', default='airplane', help='training category')
        self.parser.add_argument('-data_path', default='./data/cifar', help='path to data')
        self.parser.add_argument('-output_dir', default='./result_images', help='directory to save output synthesized images')
        self.parser.add_argument('-log_dir', default='./checkpoint', help='directory to save logs')
        self.parser.add_argument('-ckpt_dir', default='./checkpoint', help='directory to save checkpoints')
        self.parser.add_argument('-log_epoch', type=int, default=50, help='save checkpoint each `log_epoch` epochs')

        self.parser.add_argument('-set', default='scene', help='which dataset, scene/cifar/lsun/svhn')

        #Generator Parameters
        self.parser.add_argument('-ckpt_gen', default=None, help='load checkpoint for generator')
        self.parser.add_argument('-sigma_gen', type=float, default=0.3,help='sigma of reference distribution')
        self.parser.add_argument('-step_num_gen', type=int, default=2000, help='generator optimization maximum steps')
        self.parser.add_argument('-lr_gen', type=float, default=1e-3,help='learning rate of generator')
        self.parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')

        #Descriptor Parameters
        self.parser.add_argument('-ckpt_dis', default=None, help='load checkpoint for discriminator')
        self.parser.add_argument('-lr_dis', type=float, default=1e-4,help='learning rate of discriminator')
        self.parser.add_argument('-beta1_dis', type=float, default=0.5,help='beta of Adam for discriminator')
        self.parser.add_argument('-langevin_step_size_dis', type=float, default=0.001, help='langevin step size for discriminator')
        self.parser.add_argument('-sigma_dis', type=float, default=0.016,help='sigma of reference distribution')

        #Syhthethic Parameters
        self.parser.add_argument('-lr_s', type=float, default=1e-3,help='learning rate of synthetic')
        self.parser.add_argument('-beta1_s', type=float, default=0.5,help='beta of Adam for synthetic')

        #Test load models
        self.parser.add_argument('-dis_model', default='./checkpoint/discriminator_300', help='directory to save checkpoints')
        self.parser.add_argument('-gen_model', default='./checkpoint/generator_300', help='directory to save checkpoints')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.ckpt_dir):
            os.makedirs(self.opt.ckpt_dir)
        file_name = os.path.join(self.opt.ckpt_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        return self.opt
