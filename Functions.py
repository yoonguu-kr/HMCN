import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools
import os
import re
from datetime import datetime
import logging
from pynvml.smi import nvidia_smi
from config import _C as cfg
import random
import warnings
import sys
import pynvml
from Data_augmentation import get_transform, get_transform_test




class best_dc:
    '''
    save the best_dc value class
    '''
    def __init__(self):
        self.best_dice = 0
        self.best_dice_epoch=0
    def best_memory(self, dice_score, epoch):
        '''
        if input parameter dice_score is best dice score, it turns True
        :param dice_score:
        :return:
        '''
        if self.best_dice <= dice_score:
            self.best_dice = dice_score
            self.best_dice_epoch = epoch
            print(f"self.best_dice : {self.best_dice} at epoch :{self.best_dice_epoch}")
            return True
        else:
            return False

    def get_best_dc(self):
        return self.best_dice, self.best_dice_epoch

def dice(im1, atlas, labels=None):
    if labels is None:
        unique_class = np.unique(atlas)
    else:
        unique_class = labels.copy()
    dice = 0
    num_count = 0
    eps = 1e-7
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / (num_count + eps)

def dice_multi(vol1, vol2, num_classes=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    num_classes : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    # print('hello')
    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background
    # print(f'len(num_classes) : {len(num_classes)}')
    # print('hello1')
    dicem = np.zeros(len(num_classes))
    dicem2 = np.zeros(len(num_classes))
    # print('hello2')
    for idx, lab in enumerate(num_classes):
        # print(f'idx : {idx} lab : {lab}')
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2. * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = 1.0 * top / bottom
        dicem2[idx] = 1.0 * int(np.sum(vol1l)) / (vol1l.shape[2] * vol1l.shape[3] * vol1l.shape[4])
    # print('hello3')
    if nargout == 1:
        return dicem, dicem2
    else:
        return (dicem, dicem2, num_classes)


# A = best_dc()
# print("A.get_best_dc() : ", A.get_best_dc())
# A.best_memory(0.5, 1)
# result_A1 = A.get_best_dc()
# A.get_best_dc()[0]
# for i in range(1,5):
#     print(f'i : {i}')
#     best_TF = A.best_memory(A.get_best_dc()[0] * 1.1, i)
#
#     print("A.get_best_dc() : ", A.get_best_dc())
#     print(f"best_TF : {best_TF}\n ")




def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def cuda_check(logger_config):
    if not torch.cuda.is_available():
        assert torch.cuda.is_available(), "torch cuda is not available, Please configure the gpu first"
    else:
        #   TO USE pynvml.nvmlSystemGetDriverVersion() function
        pynvml.nvmlInit()
        logger_config.info("=" * 50)
        logger_config.info("This is running in GPU")
        logger_config.info("torch.cuda.is_available() : {}".format(torch.cuda.is_available()))
        logger_config.info("torch.cuda.is_available() : {}".format(torch.cuda.device_count()))
        logger_config.info("Driver Version : {}".format(pynvml.nvmlSystemGetDriverVersion()))

        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            logger_config.info("Device {} : {}".format(i, pynvml.nvmlDeviceGetName(handle)))
        nvsmi = nvidia_smi.getInstance()
        logger_config.info("=" * 50+'\n')
        # GPU 할당 변경하기
        # 원하는 GPU 번호 입력

        device = torch.device(f'cuda:{cfg.MODEL.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)  # change allocation of current GPU
        logger_config.info('Current cuda device '.format(torch.cuda.current_device()))
        logger_config.info('Using device : {} >>{}'.format(device, torch.cuda.get_device_name(device)))
        logger_config.info(device.type)

        if device.type == 'cuda':
            logger_config.info(torch.cuda.get_device_name(cfg.MODEL.GPU_NUM))
            logger_config.info('Memory Usage:')
            logger_config.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(cfg.MODEL.GPU_NUM) / 1024 ** 3, 1)))
            logger_config.info('Reserved:  {} GB'.format(round(torch.cuda.memory_reserved(device) / 1024 ** 3, 1)))
            logger_config.info(torch.cuda.get_device_properties('cuda:{}'.format(cfg.MODEL.GPU_NUM)))

        elif device.type == 'cpu':
            assert device.type != 'cpu', "device is running in cpu, please configure it first"


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory : ", directory)

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')


def tr_val_test(names, tr_val_ratio, test=False):
    '''
    if test is False
     --> training data(ratio:8), validation data(ratio:2)
    else test is True
     --> training data(ratio:8), validation data(ratio:2 - 2), test_check_data1, test_check_data2
    :param names: list of names
    :param tr_val_ratio:
    :return:
    '''
    if test:
        if int(len(names) * (1-tr_val_ratio)) >=2:
            num_tr = int(len(names) * tr_val_ratio)
            names_tr = names[:num_tr]
            names_test1 = names[num_tr]
            names_test2 = names[num_tr+1]
            names_val= names[num_tr+2:]
            return names_tr, names_val, names_test1, names_test2
        else:
            raise Exception("Test data number is less than 2")
    else:
        # raise Exception("This only results Tr, val and 2 test to run this code")
        num_tr = int(len(names) * tr_val_ratio)
        names_tr = names[:num_tr]
        names_val = names[num_tr:]
        return names_tr, names_val



def fileNameUpdate(directory, *argv, dateupdate=False, num_version=None):
    '''
    This is making a file name
    " name_prefix +"_YYMMDD" +"fold"+"_version"."ext"
    - ex)
        " name_prefix +"_YYMMDD" + "_v1"
        " name_prefix +"_YYMMDD" + "_v2"
        " name_prefix +"_YYMMDD" + "_v3"

    :param directory: directory
    :param dateupdate: whether YYMMDD is added or not
    :param num_version: version number
    :return: filename_YYMMDD_version.ext
    '''

    filename_check =''
    for arg in argv:
        filename_check+= str(arg)
    # print("filename_check : ", filename_check)

    # files = [file for file in os.listdir(directory) if filename_check in file]
    files = [file for file in os.listdir(directory) if filename_check in file and file[:file.rindex("_")] == filename_check]

    #To check the version name
    if num_version:
        file_version = num_version
    else:
        if files:
            file_version = max([int(re.findall(r'\d+', file)[-1]) for file in files if filename_check in file]) + 1
        else:
            file_version = 1

    now = datetime.now()
    current_time = now.strftime("%Y%m%d")[2:]

    if dateupdate:
        filename = filename_check+"_"+current_time+"_v"+str(file_version)
    else:
        filename= filename_check+"_v"+str(file_version)

    return filename, file_version


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)/2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)/2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)/2

    return flow

def load_3D(name):
    X = nib.load(name)
    X = X.get_fdata()
    return X


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

        self.avg = self.sum / self.count


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()

class Dataset_epoch_check(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        print(f'self.names : {self.names}')
        print(f'len(self.names) : {len(self.names)}')
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        print(f'len(self.index_pair) : {len(self.index_pair)}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()



class Dataset_epoch_validation(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, imgs, labels, norm=False):
        'Initialization'
        super(Dataset_epoch_validation, self).__init__()



        self.imgs = imgs
        self.labels = labels
        self.norm = norm
        self.imgs_pair = list(itertools.permutations(imgs, 2))
        self.labels_pair = list(itertools.permutations(labels, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_pair)

    def __getitem__(self, step):
        # print(f'Dataset_epoch_validation --step : {step}')
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.imgs_pair[step][0])
        img_B = load_4D(self.imgs_pair[step][1])

        label_A = load_4D(self.labels_pair[step][0])
        label_B = load_4D(self.labels_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()


class Dataset_Mindboggle(Data.Dataset):
    '''
    Data loader for Mindboggle
    #isTest == False -> for Training
    #isTest == True -> for test
    '''

    def __init__(self, names, norm=False, img_size=(72, 96, 72), isTest=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.img_size = img_size

        if isTest:
            self.transform = get_transform_test(img_size=self.img_size)
            self.constrain = list(range(1, 43, 1))
        else:
            self.transform = get_transform(img_size=self.img_size)
            self.constrain = list(range(43, 63, 1))

        self.img_paths = []
        for file in self.names:
            file_name = os.path.basename(os.path.normpath(file))
            if (file_name.endswith(".nii.gz") and file_name.startswith("brain")):
                for c in self.constrain:
                    if "brain_{0:02d}".format(c) in str(file_name) :
                        break
                else:
                    self.img_paths.append(file)

        self.index_pair = list(itertools.permutations(self.img_paths, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        self.img_mov_path = self.index_pair[step][0]
        self.img_fix_path = self.index_pair[step][1]

        img_mov = load_3D(self.index_pair[step][0])
        img_fix = load_3D(self.index_pair[step][1])

        self.img_mov_atlas_path = self.img_mov_path.replace("brain", f"atlas")
        self.img_fix_atlas_path = self.img_fix_path.replace("brain", f"atlas")

        img_mov_atlas = load_3D(self.img_mov_atlas_path)
        img_fix_atlas = load_3D(self.img_fix_atlas_path)

        # print(f"img_mov_path : {self.img_mov_path}")
        # print(f"img_mov_atlas_path : {self.img_mov_atlas_path}")
        # print(f"img_fix_path : {self.img_fix_path}")
        # print(f"img_fix_atlas_path : {self.img_fix_atlas_path}")
        # print()
        fixed_img_pytorch, moving_img_pytorch, img_fix_atlas_pytorch, img_mov_atlas_pytorch = self.transform(
            [img_fix, img_mov, img_fix_atlas, img_mov_atlas])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch

class Dataset_LPBA40(Data.Dataset):
    'Characterizes a dataset for PyTorch - for training'

    def __init__(self, names, img_size=(72, 96, 72), norm=False):
        '''
        Initialization
        :param names: file name list
        :param norm: even tho there is varaible norm,
                    but we only use the normalized LPBA40 files through preprocessing code.
                    so we don't use this variable
        '''
        self.names = names
        self.norm = norm
        self.img_size = img_size

    def initialize(self):
        '''
        to make the list of moving_paths(moving), dictionary of moving_fixed(fixed)
            dictionary of moving_fixed(fixed) --> {moving_path(key) : moving_fix(value)}
`
        :param is_training: bool--> to make train dataloader, validation dataloader
        :return:
        '''



        self.constrain = list(range(31, 41, 1))
        self.transform = get_transform(img_size=self.img_size)


        self.moving_paths = []
        for file in self.names:
            file_name = os.path.basename(os.path.normpath(file))
            if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                for c in self.constrain:
                    if "l{}_".format(str(c)) in str(file_name) or "_l{}.".format(str(c)) in str(file_name):
                        break
                else:
                    self.moving_paths.append(file)

        self.fixed_paths = {}
        self.moving_atlas_paths = {}
        self.fixed_atlas_paths = {}
        for file_moving in self.moving_paths:
            path_parents = os.path.abspath(os.path.join(file_moving, os.pardir))
            file_name_moving = os.path.basename(os.path.normpath(file_moving))
            fixed_name = file_name_moving.split(".")[-2].split("_")[-1]
            fixed_suffix = file_name_moving.split(".")[-1]

            if not (fixed_suffix == "hdr" or fixed_suffix == "nii"):
                raise Exception("Suffix not hdr or nii.")
            self.fixed_paths[file_moving] = os.path.join(path_parents,
                                                          "{}_to_{}.{}".format(str(fixed_name), str(fixed_name),
                                                                               str(fixed_suffix)))
            self.moving_atlas_paths[file_moving] = file_moving.replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")
            self.fixed_atlas_paths[file_moving] = self.fixed_paths[file_moving].replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fixed_paths)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        dir_moving = self.moving_paths[step]
        dir_fix = self.fixed_paths[self.moving_paths[step]]

        img_mov = load_3D(dir_moving) # type numpy
        img_fix = load_3D(dir_fix) # type numpy

        dir_moving_atlas = self.moving_atlas_paths[self.moving_paths[step]]
        dir_fixed_atlas = self.fixed_atlas_paths[self.moving_paths[step]]

        img_mov_atlas = load_3D(dir_moving_atlas)  # type numpy
        img_fix_atlas = load_3D(dir_fixed_atlas)  # type numpy

        if self.norm:
            img_mov = imgnorm(img_mov)
            img_fix = imgnorm(img_fix)

        # img_mov_torch = torch.from_numpy(img_mov).float()
        # img_fix_torch = torch.from_numpy(img_fix).float()
        # print(f'img_mov_torch.shape : {img_mov_torch.shape}')
        # print(f'img_fix_torch.shape : {img_fix_torch.shape}')
        # img_mov_atlas_torch = torch.from_numpy(img_mov_atlas).float()
        # img_fix_atlas_torch = torch.from_numpy(img_fix_atlas).float()

        #self.transform_train is change based on is_training variable
        moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch = self.transform(
            [img_mov, img_fix, img_mov_atlas, img_fix_atlas])
        # print(f'type(img_mov) : {type(img_mov)}') #class 'numpy.memmap
        # print(f'img_mov.shape : {img_mov.shape}') #(80, 106, 80)
        # print(f'type(moving_img_pytorch) : {type(moving_img_pytorch)}') #class 'torch.Tensor'
        # print(f'moving_img_pytorch.shape : {moving_img_pytorch.shape}') #torch.Size([1, 72, 96, 72])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch


class Dataset_LPBA40_ValTest(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, img_size=(72, 96, 72), norm=False):
        '''
        Initialization
        :param names: file name list
        :param norm: even tho there is varaible norm,
                    but we only use the normalized LPBA40 files through preprocessing code.
                    so we don't use this variable
        '''
        self.names = names
        self.norm = norm
        self.img_size = img_size

    def initialize(self, is_valid):
        '''
        to make the list of moving_paths(moving), dictionary of moving_fixed(fixed)
            dictionary of moving_fixed(fixed) --> {moving_path(key) : moving_fix(value)}
`
        :param is_training: bool--> to make train dataloader, validation dataloader
        :return:
        '''
        self.is_valid = is_valid

        self.transform = get_transform_test(img_size=self.img_size)
        if self.is_valid =='test':
            self.constrain = list(range(1, 31, 1))


            self.moving_paths = []
            for file in self.names:
                file_name = os.path.basename(os.path.normpath(file))
                if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                    for c in self.constrain:
                        if "l{}_".format(str(c)) in str(file_name) or "_l{}.".format(str(c)) in str(file_name):
                            break
                    else:
                        self.moving_paths.append(file)

        elif self.is_valid=='val':
            constrain1 = list(range(1, 31, 1))
            constrain2 = list(range(31, 41, 1))

            self.moving_paths = []
            for file in self.names:
                file_name = os.path.basename(os.path.normpath(file))
                if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                    for c1 in constrain1:
                        for c2 in constrain2:
                            if "l{}_".format(str(c1)) in str(file_name) and "_l{}.".format(str(c2)) in str(file_name):
                                self.moving_paths.append(file)
                            if "l{}_".format(str(c2)) in str(file_name) and "_l{}.".format(str(c1)) in str(file_name):
                                self.moving_paths.append(file)

        # self.moving_fixed = {}
        # self.moving_atlases = {}
        # self.fixed_atlases = {}


        self.fixed_paths = {}
        self.moving_atlas_paths = {}
        self.fixed_atlas_paths = {}
        for file_moving in self.moving_paths:
            path_parents = os.path.abspath(os.path.join(file_moving, os.pardir))
            file_name_moving = os.path.basename(os.path.normpath(file_moving))
            fixed_name = file_name_moving.split(".")[-2].split("_")[-1]
            fixed_suffix = file_name_moving.split(".")[-1]

            if not (fixed_suffix == "hdr" or fixed_suffix == "nii"):
                raise Exception("Suffix not hdr or nii.")
            self.fixed_paths[file_moving] = os.path.join(path_parents,
                                                          "{}_to_{}.{}".format(str(fixed_name), str(fixed_name),
                                                                               str(fixed_suffix)))
            self.moving_atlas_paths[file_moving] = file_moving.replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")
            self.fixed_atlas_paths[file_moving] = self.fixed_paths[file_moving].replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")



        # print(f'len(fixed_paths) : {len(self.fixed_paths)}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fixed_paths)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        dir_moving = self.moving_paths[step]
        dir_fix = self.fixed_paths[self.moving_paths[step]]

        img_mov = load_3D(dir_moving) # type numpy
        img_fix = load_3D(dir_fix) # type numpy

        dir_moving_atlas = self.moving_atlas_paths[self.moving_paths[step]]
        dir_fixed_atlas = self.fixed_atlas_paths[self.moving_paths[step]]

        img_mov_atlas = load_3D(dir_moving_atlas)  # type numpy
        img_fix_atlas = load_3D(dir_fixed_atlas)  # type numpy

        if self.norm:
            img_mov = imgnorm(img_mov)
            img_fix = imgnorm(img_fix)

        # img_mov_torch = torch.from_numpy(img_mov).float()
        # img_fix_torch = torch.from_numpy(img_fix).float()
        #
        # img_mov_atlas_torch = torch.from_numpy(img_mov_atlas).float()
        # img_fix_atlas_torch = torch.from_numpy(img_fix_atlas).float()

        #self.transform_train is change based on is_training variable
        img_mov_pytorch, img_fix_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch = self.transform(
            [img_mov, img_fix, img_mov_atlas, img_fix_atlas])

        return img_mov_pytorch, img_fix_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output

