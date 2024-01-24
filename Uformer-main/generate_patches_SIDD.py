from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

'''argparse 模块用于为你的脚本或程序创建命令行接口。它允许你定义脚本接受的参数，指定它们的类型，
并提供帮助信息。当用户从命令行运行你的脚本时，argparse 帮助解析命令行参数并提供用户友好的界面。
'''
#argparse.ArgumentParser：创建一个参数解析器对象。该对象将包含有关你的脚本可以接受的参数的信息。
#description='Generate patches from Full Resolution images'：提供了脚本或程序的简要描述。
# 当用户使用 --help 选项运行你的脚本时，这个描述将显示出来，帮助用户了解脚本的功能。
parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
'''
一下的add_argument方法用于给parser添加一个参数，参数的名字就是括号里的第一个参数，如：(--src_dir) ，default是这个参数的默认值，type表示参数要
解释为一个字符串，help是用户调用--help时会显示的信息
'''
parser.add_argument('--src_dir', default='../SIDD_Small_sRGB_Only/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/denoising/sidd/train',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=300, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=4, type=int, help='Number of CPU Cores')

#解析命令行参数
args = parser.parse_args()

#获取对应的参数的值，比如parser.ser_dir就是上面对应的‘--src_dir’的值
src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

#把两个路径连接起来，形成新的路径
noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'groundtruth')

#如果tar目录存在，则删除
#if os.path.exists(tar):
 #   os.system("rm -r {}".format(tar))

#创建一个目录
os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

#get sorted folders
files = natsorted(glob(os.path.join(src, '*', '*.PNG')))

#获取噪声和纯净的文档名字列表
noisy_files, clean_files = [], []
for file_ in files:
    filename = os.path.split(file_)[-1]
    if 'GT' in filename:
        clean_files.append(file_)
    if 'NOISY' in filename:
        noisy_files.append(file_)

def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    #返回表示图像的numpy数组
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        #图像图像的局部补丁信息，参数分别是H,W和RGB通道(默认红绿蓝)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        #重写信息
        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)



Parallel(n_jobs=1)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))  #采用多线程时我会报错，就是在n_jobs>1时