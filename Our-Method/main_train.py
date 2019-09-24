from model.model import Joint
from opts import opts

def main():
    opt=opts().parse()
    model=Joint(opt)
    model.train()

if __name__=='__main__':
    main()
