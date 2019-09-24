from model.model import WINN
from opts import opts
import torch

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
