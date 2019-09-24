from model.model import Joint
from opts import opts

def main():
    opt=opts().parse()
    
    model=Joint(opt, discriminator_model=opt.dis_model, generator_model=opt.gen_model)
    
    model.test()

if __name__=='__main__':
    main()
