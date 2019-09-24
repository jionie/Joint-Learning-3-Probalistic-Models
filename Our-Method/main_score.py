from inception_score import *
from inception_model import *
from frechet_inception_distance import *
from opts import *
from utils.data_io import *
import torchvision
import torchvision.transforms as transforms
import torch



if __name__=='__main__':

    opt = opts().parse()

    num_train = 1000
    num_test = 1000
    if opt.set == 'scene' or opt.set == 'cifar' or opt.set == 'svhn':
        train_data = DataSet(os.path.join(opt.data_path, opt.category), 
                                    image_size=opt.img_size)[:num_train]
    else:
        train_data = torchvision.datasets.LSUN(root=opt.data_path,
                                    classes=['bedroom_train'],
                                    transform=transforms.Compose([transforms.Resize(opt.img_size),
                                    transforms.ToTensor(), ]))[:num_train]

    Real_image_tensor = torch.tensor(np.array(train_data), dtype=torch.float)

    generate_image_root = os.path.join(opt.output_dir, "generate")
    Generated_data = DataSet(generate_image_root, image_size=opt.img_size)[:num_test]
    Generated_image_tensor = torch.tensor(np.array(Generated_data), dtype=torch.float)


    Revised_image_root = os.path.join(opt.output_dir, "des")
    Revised_data = DataSet(Revised_image_root, image_size=opt.img_size)[:num_test]
    Revised_image_tensor = torch.tensor(np.array(Revised_data), dtype=torch.float)

    print("Inception score (mean, std) for real dataset: ", inception_score(Real_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))
    print("Inception score (mean, std) for genrated dataset: ", inception_score(Generated_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))
    print("Inception score (mean, std) for revised generated dataset: ", inception_score(Revised_image_tensor, cuda=True, batch_size=32, resize=True, splits=10))


    Real_Generated_fid_value = calculate_fid_given_paths(Real_image_tensor, Generated_image_tensor, batch_size=50, cuda=True, dims=2048)
    Real_Revised_fid_value = calculate_fid_given_paths(Real_image_tensor, Revised_image_tensor, batch_size=50, cuda=True, dims=2048)

    print("Frechet Inception Distance between real dataset and generated dataset: ", Real_Generated_fid_value)
    print("Frechet Inception Distance between real dataset and revised generated dataset: ", Real_Revised_fid_value)
