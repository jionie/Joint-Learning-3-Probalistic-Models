"""
Usage： python3 download-trinity.py {NameOfDataset}

The download script will download the dataset and transform them into formats that is readable  

Currently LSUN is undownloadable but can be directly used from PyTorch

Modification of https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
Downloads the following:
- Imagenet-Scene dataset
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
- CIFAR dataset
- SVHN dataset
"""

from __future__ import print_function
import os
import sys
import json
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib
import cv2
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Download dataset for CoopNet.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['place205', 'scene', 'celebA', 'lsun', 'mnist','cifar','SVHN'],
           help='name of dataset to download [celebA, lsun, mnist]')

def download(url, dirpath):
  filename = url.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  u = urllib.request.urlopen(url)
  f = open(filepath, 'wb')
  filesize = int(u.headers["Content-Length"])
  print("Downloading: %s Bytes: %s" % (filename, filesize))

  downloaded = 0
  block_sz = 8192
  status_width = 70
  while True:
    buf = u.read(block_sz)
    if not buf:
      print('')
      break
    else:
      print('', end='\r')
    downloaded += len(buf)
    f.write(buf)
    status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
    print(status, end='')
    sys.stdout.flush()
  f.close()
  return filepath

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)

def download_imagenet_scene(dirpath):
    data_dir = 'scene'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Scene - skip')
        return
    filename, drive_id = "imagenet_scene.zip", "1RZ2zfYhoq714uvlan7V8mHeUb9XpJqWq"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)
    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(save_path)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def download_place205(dirpath):
  data_dir = os.path.join(dirpath, 'place205')
  if os.path.exists(data_dir):
    print('Found Place205 - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://data.csail.mit.edu/places/places205/'
  filename = 'imagesPlaces205_resize.tar.gz'
  url = (url_base+filename).format(**locals())
  out_path = os.path.join(data_dir, filename)
  cmd = ['curl', url, '-o', out_path]
  print('Downloading ', filename)
  subprocess.call(cmd)
  unzip(out_path)


def download_celeb_a(dirpath):
  data_dir = 'celebA'
  if os.path.exists(os.path.join(dirpath, data_dir)):
    print('Found Celeb-A - skip')
    return

  filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
  save_path = os.path.join(dirpath, filename)

  if os.path.exists(save_path):
    print('[*] {} already exists'.format(save_path))
  else:
    download_file_from_google_drive(drive_id, save_path)

  zip_dir = ''
  with zipfile.ZipFile(save_path) as zf:
    zip_dir = zf.namelist()[0]
    zf.extractall(dirpath)
  os.remove(save_path)
  os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def _list_categories(tag):
  url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
  f = urllib.request.urlopen(url)
  return json.loads(f.read())

def _download_lsun(out_dir, category, set_name, tag):
  url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
      '&category={category}&set={set_name}'.format(**locals())
  print(url)
  if set_name == 'test':
    out_name = 'test_lmdb.zip'
  else:
    out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
  out_path = os.path.join(out_dir, out_name)
  cmd = ['curl', url, '-o', out_path]
  print('Downloading', category, set_name, 'set')
  subprocess.call(cmd)

def download_lsun(dirpath):
  data_dir = os.path.join(dirpath, 'lsun')
  if os.path.exists(data_dir):
    print('Found LSUN - skip')
    return
  else:
    os.mkdir(data_dir)

  tag = 'latest'
  categories = _list_categories(tag)

  for category in categories:
    _download_lsun(data_dir, category, 'train', tag)
    _download_lsun(data_dir, category, 'val', tag)
  _download_lsun(data_dir, '', 'test', tag)

def download_mnist(dirpath):
  data_dir = os.path.join(dirpath, 'mnist')
  if os.path.exists(data_dir):
    print('Found MNIST - skip')
    return
  else:
    os.mkdir(data_dir)
  url_base = 'http://yann.lecun.com/exdb/mnist/'
  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir,file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)
  train_image = 'train-images-idx3-ubyte'
  train_label = 'train-labels-idx1-ubyte'
  test_image = 't10k-images-idx3-ubyte'
  test_label = 't10k-labels-idx1-ubyte'
  for image_f, label_f in [(train_image, train_label), (test_image, test_label)]:
    with open(os.path.join(data_dir,image_f), 'rb') as f:
      images = f.read()
    with open(os.path.join(data_dir,label_f), 'rb') as f:
      labels = f.read()  
    images = [d for d in images[16:]]
    images = np.array(images, dtype=np.uint8)
    images = images.reshape((-1,28,28))
    if image_f == train_image:
      outdir = os.path.join(data_dir,'train')
    else:
      outdir = os.path.join(data_dir,'test')
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    for k,image in enumerate(images):
      cv2.imwrite(os.path.join(outdir, '%05d.png' % (k,)), image)
  
    '''labels = [outdir + '/%05d.png %d' % (k, l) for k,l in enumerate(labels[8:])]
    with open('%s.txt' % label_f, 'w') as f:
      f.write(os.linesep.join(labels))'''
def download_cifariz(dirpath):
  data_dir = os.path.join(dirpath, 'cifar')
  if os.path.exists(data_dir):
    print('Found cifar - skip')
    return
  cmd = ['pip', 'install','cifar2png']
  subprocess.call(cmd)
  cmd = ['cifar2png', 'cifar10',data_dir,'--name-with-batch-index']
  subprocess.call(cmd)
def download_cifar(dirpath):
  data_dir = os.path.join(dirpath, 'cifar')
  if os.path.exists(data_dir):
    print('Found cifar - skip')
    return
  else:
   os.mkdir(data_dir)
  url_base = 'https://www.cs.toronto.edu/~kriz/'
  file_name='cifar-10-python.tar.gz'
  url = (url_base+file_name).format(**locals())
  print(url)
  out_path = os.path.join(data_dir,file_name)
  cmd = ['curl', url, '-o', out_path]
  print('Downloading ', file_name)
  subprocess.call(cmd)
  cmd = ['gzip', '-d', out_path]  
  print('Decompressing ', file_name)
  subprocess.call(cmd)
  cmd = ['tar', 'xf', out_path[:-3],'--directory',data_dir]
  subprocess.call(cmd)
  files=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
  cate=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  for file in files:
    f = open(data_dir+'/cifar-10-batches-py/'+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    #print(dict.keys())
    images = dict[b'data']
    #images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    images = np.array(images)   #   (10000, 3072)
    lab = np.array(labels)   #   (10000,)
    images = np.reshape(images, (10000, 3, 32, 32))
    for i in cate:
      if not os.path.exists(os.path.join(data_dir,i)):
        os.mkdir(os.path.join(data_dir,i))
    for i, img in enumerate(images):
      img=img.transpose(1,2,0) # cv2 needs 32x32x3, the img n cifar is 3x32x32
      img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  #cv2 needs BGR, cifar is RGBS
      cv2.imwrite(data_dir+'/'+cate[lab[i]]+'/'+file+'_'+str(i)+'.png', img)     # image no #
def download_svhn(dirpath):
  data_dir = os.path.join(dirpath, 'SVHN')
  if os.path.exists(data_dir):
    print('Found SVHN - skip')
    return
  else:
   os.mkdir(data_dir)
  url_base = 'http://ufldl.stanford.edu/housenumbers/'
  file_names = ['train.tar.gz',
                'test.tar.gz',
                'extra.tar.gz']
  for file_name in file_names:
    url = (url_base+file_name).format(**locals())
    print(url)
    out_path = os.path.join(data_dir,file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)
    cmd = ['tar', 'xf', out_path[:-3],'--directory',data_dir]
    subprocess.call(cmd)
def prepare_data_dir(path = './data'):
  if not os.path.exists(path):
    os.mkdir(path)

if __name__ == '__main__':
  args = parser.parse_args()
  prepare_data_dir("./data")
  if any(name in args.datasets for name in ['place205']):
      download_place205('./data')
  if any(name in args.datasets for name in ['scene']):
      download_imagenet_scene('./data')
  if any(name in args.datasets for name in ['CelebA', 'celebA', 'celebA']):
    download_celeb_a('./data')
  if 'lsun' in args.datasets:
    download_lsun('./data')
  if 'mnist' in args.datasets:
    download_mnist('./data')
  if 'cifar' in args.datasets:
    download_cifar('./data')
  if 'SVHN' in args.datasets:
    download_svhn('./data')