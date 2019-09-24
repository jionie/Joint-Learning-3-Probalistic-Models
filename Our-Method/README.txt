# ECE 6254 Team 32 Project

To run our project locally:

(1)First, follow the README_download to run download.py to download dataset.

(2)Second, follow the README_train to run main_train.py to train models.

(3)Third, follow the README_test to run main_test.py to do inference and get images.

(3)Fourth, follow the README_score to run main_score.py to get inception score and fid for current models.


*****************************************************************
*********************Requirements*****************************
pytorch==0.4.0 
torchvision
numpy


*****************************************************************
**********************README_download*****************************
Run "python3 download.py datasetname" to download datasets including (imagenet/scene, LSUN, Cifar10, SVHN)
datasetname: scene, lsun, cifar, SVHN
For lsun, we only download bedroom dataset, your can change variable categories in function download_lsun(dirpath) to download more categories


*****************************************************************
**********************README_train*****************************
Your can simply change the dataset and parameters in opts.py then just simply run "python3 main_train.py"
Training set will contain num_train real images, default num_train=1000
All the images and models will be saved every 100 iterations


*****************************************************************
**********************README_test*****************************
Your can simply change the dataset, dis_model and gen_model to test in opts.py then just simply run "python3 main_test.py"
You're get num_test generated images and revised generated images, default num_test=1000


*****************************************************************
**********************README_score*****************************
Your can simply calculate inception score and fid for generated images, revised generated images and real training images by running "python3 main_score.py"
default num_train=1000, num_test=1000


