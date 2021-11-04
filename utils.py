# Imports
# from PIL import Image
from utils import *

import gc
import pickle
from collections import OrderedDict

import numpy as np
from scipy import linalg
import torch.nn
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import Dataset

# from vit_my_utils import *
# from dn2_vit_experiment import *
from os.path import  join
import os

import pickle
from pytorch_pretrained_vit.model import AnomalyViT
import bisect
import pandas as pd
from numpy.linalg import eig

# sys.path.insert(0, '/home/access/thesis/anomaly_detection/code/Unsupervised-Classification/')
# from Pretrained_ViT_scoring_method_main import *

from sklearn import mixture
import argparse
# from scoring_methods import GMM
# import AnomalyResNet

from utils import *

import matplotlib.pyplot as plt
from enum import Enum
import os
import time
import sys

import os
from os.path import join

import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder

import numpy as np
import math


import sys
import json
# from PIL import Image
# from visualizer import *
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder
from torchvision.transforms import *
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_vit import ViT
import faiss
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import pandas as pd
import glob
import os
import logging
import random
from multiprocessing import Process
import transform_layers as TL
from feature_extactor import *
#Aux.
from os import listdir
from os.path import isfile, join
from PIL import Image

class DiorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 image_path,
                 labels_dict_path,
                 transform=None):
        """
        Args:
            image_path (string): Path to the images.
            labels_dict_path (string): Path to the dict with annotations.
        """
        self.image_path = image_path
        self.labels_dict_path = labels_dict_path
        self.transform = transform

        with open(self.labels_dict_path, 'rb') as handle:
            self.labels_dict = pickle.load(handle)
        self.images =  [f for f in listdir(image_path) if isfile(join(image_path, f))]
        self.targets = [self.labels_dict[img]['label_index'] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(join(self.image_path, self.images[idx]))
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_simclr_augmentation(resize_factor = 0.99, image_size=224):

    # parameter for resizecrop
    resize_scale = (resize_factor, 1.0) # resize scaling factor
    # if P.resize_fix: # if resize_fix is True, use same scale
    #     resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    # color_jitter = TL.ColorJitterLayer(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8)
    color_jitter = TL.ColorJitterLayer(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=0.8)

    # color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)


    transform = nn.Sequential(
        color_jitter,
        # color_gray,
        resize_crop,
    )

    return transform

from torchvision.transforms import *
def get_my_augmentations(is_fmnist):
    if is_fmnist:
        transforms = Compose([
            ToPILImage(),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ToTensor(),
            # Normalize(**_normalize),
            # ToTensor()
        ]
        )

    else:
        transforms = Compose([
            ToPILImage(),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ToTensor(),
            # Normalize(**_normalize),
            # ToTensor()
        ]
        )

    return transforms

def get_features(model, dataLoader, early_break=-1, is_fmnist = False):

    clustered_features = []
    # simclr_aug = get_simclr_augmentation()
    simclr_aug =get_my_augmentations(is_fmnist = is_fmnist)
    for i, (data, _) in enumerate(tqdm(dataLoader)):
        if early_break > 0 and early_break < i:
            break

        encoded_outputs = model(data.to('cuda'))
        clustered_features.append(encoded_outputs.detach().cpu().numpy())

    clustered_features = np.concatenate(clustered_features)
    return clustered_features




def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def extract_fetures(base_path,
                    datasets,
                    model,
                    logging,
                    calculate_features=False,
                    manual_class_num_range =None,
                    few_shot_n_samples = -1,
                    one_vs_rest_vals = [True, False],
                    output_train_features = True,
                    output_test_features = True,
                    use_imagenet = False,
                    ):
    exp_num = -1


    for dataset in datasets:
        print_and_add_to_log("=======================================", logging)
        print_and_add_to_log(f"Dataset: {dataset}", logging)
        print_and_add_to_log(f"Path: {base_path}", logging)
        print_and_add_to_log("=======================================", logging)
        exp_num += 1
        if dataset == 'cifar100':
            _class_num = 20

        elif dataset == 'cats_vs_dogs':
            _class_num = 2

        elif dataset == 'dior':
            _class_num = 19

        elif dataset == 'wbc':
            _class_num = 4

        elif dataset == 'flowers':
            _class_num = 20

        # additional datasets
        elif dataset == 'blood_cells':
            _class_num = 4

        elif dataset == 'covid':
            _class_num = 2

        elif dataset == 'intel_classification':
            _class_num = 6

        elif dataset == 'weather_recognition':
            _class_num = 4

        elif dataset == 'concrete_crack_classification':
            _class_num = 2

        else:
            _class_num = 10

        if manual_class_num_range is not None:
            _classes = range(*manual_class_num_range)

        else:
            _classes = range(_class_num)


        for _class in [0]:

            # config
            for one_vs_rest in one_vs_rest_vals:

                print_and_add_to_log("================================================================", logging)
                print_and_add_to_log(f"Experiment number: {exp_num}", logging)
                print_and_add_to_log(f"Dataset: {dataset}", logging)
                print_and_add_to_log(f"Class: {_class}", logging)
                print_and_add_to_log(f"One vs rest: {one_vs_rest}", logging)

                if dataset == 'cifar10':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/cifar10/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/cifar10/class_{str(_class)}')

                    number_of_classes = 10
                    _normalize = {
                        'mean': [0.4914, 0.4822, 0.4465],
                        'std': [0.2023, 0.1994, 0.201]

                    }


                elif dataset == 'cifar100':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/cifar100/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/cifar100/class_{str(_class)}')

                    number_of_classes = 20

                    _normalize = {
                        'mean': [0.5071, 0.4867, 0.4408],
                        'std': [0.2675, 0.2565, 0.2761]

                    }

                elif dataset == 'fmnist':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/fmnist/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/fmnist/class_{str(_class)}')

                    number_of_classes = 10

                elif dataset == 'cats_vs_dogs':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/catsVsDogs/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/catsVsDogs/class_{str(_class)}')

                    number_of_classes = 2

                    _normalize = {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]

                    }

                elif dataset == 'dior':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/dior/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/dior/class_{str(_class)}')

                    number_of_classes = 20

                elif dataset == 'wbc':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/wbc/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/wbc/class_{str(_class)}')

                    number_of_classes = 4

                elif dataset == 'flowers':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/flowers/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/flowers/class_{str(_class)}')

                    number_of_classes = 102

                elif dataset == 'blood_cells':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/blood_cells/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/blood_cells/class_{str(_class)}')

                    number_of_classes = 4

                elif dataset == 'covid':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/covid/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/covid/class_{str(_class)}')

                    number_of_classes = 2

                elif dataset == 'intel_classification':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/intel_classification/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/intel_classification/class_{str(_class)}')

                    number_of_classes = 6

                elif dataset == 'weather_recognition':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/weather_recognition/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/weather_recognition/class_{str(_class)}')

                    number_of_classes = 4

                elif dataset == 'concrete_crack_classification':
                    if one_vs_rest:
                        base_feature_path = join(base_path, f'1_vs_rest/concrete_crack_classification/class_{str(_class)}')
                    else:
                        base_feature_path = join(base_path, f'rest_vs_1/concrete_crack_classification/class_{str(_class)}')

                    number_of_classes = 2

                else:
                    raise ValueError(f"{dataset} not supported yet!")



                if not os.path.exists((base_feature_path)):
                    os.makedirs(base_feature_path,)
                    # logging.basicConfig(filename=join(base_feature_path, f'class_{_class}.log'),
                    #                     filemode='w', level=logging.DEBUG)
                else:
                    logging.info(f"Experiment of class {_class} already exists")

                if one_vs_rest:
                    ANOMALY_CLASSES = [i for i in range(number_of_classes) if i != _class]
                else:
                    ANOMALY_CLASSES = [_class]

                BATCH_SIZE = 18
                if dataset == 'fmnist':

                    if use_imagenet:

                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )

                else:
                    if use_imagenet:

                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )


                model.eval()
                freeze_model(model)
                model.to('cuda')

                # get dataset
                if dataset == 'cifar10':


                    trainset_origin = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                                              train=True, download=True,
                                              transform=val_transforms)

                    testset = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                                      train=False, download=True,
                                      transform=val_transforms)



                elif dataset == 'cifar100':
                    trainset_origin = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                                               train=True, download=True,
                                               transform=val_transforms)

                    testset = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                                       train=False, download=True,
                                       transform=val_transforms)

                    trainset_origin.targets = sparse2coarse(trainset_origin.targets)
                    testset.targets = sparse2coarse(testset.targets)


                elif dataset == 'fmnist':
                    trainset_origin = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                                                   train=True, download=True,
                                                   transform=val_transforms)

                    testset = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                                           train=False, download=True,
                                           transform=val_transforms)



                elif dataset == 'cats_vs_dogs':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/train',
                        transform=val_transforms)

                    testset = ImageFolder(root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/test',
                                          transform=val_transforms)


                elif dataset == 'dior':
                    trainset_origin = DiorDataset(
                        image_path='/home/access/thesis/anomaly_detection/data/dior/cropped_train_val_images/',
                        labels_dict_path='/home/access/thesis/anomaly_detection/data/dior/labels_dict.pkl',
                        transform=val_transforms)

                    testset = DiorDataset(
                        image_path='/home/access/thesis/anomaly_detection/data/dior/cropped_test_images/',
                        labels_dict_path='/home/access/thesis/anomaly_detection/data/dior/labels_dict.pkl',
                        transform=val_transforms)

                elif dataset == 'wbc':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/wbc/training',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/wbc/test',
                        transform=val_transforms)

                elif dataset == 'flowers':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/102flowers/training',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/102flowers/test',
                        transform=val_transforms)


                #additional dataset
                elif dataset == 'blood_cells':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/blood_cells/train',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/blood_cells/test',
                        transform=val_transforms)

                elif dataset == 'covid':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/covid/train',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/covid/test',
                        transform=val_transforms)

                elif dataset == 'intel_classification':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/intel_classification/train',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/intel_classification/test',
                        transform=val_transforms)

                elif dataset == 'weather_recognition':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/weather_recognition/train',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/weather_recognition/test',
                        transform=val_transforms)

                elif dataset == 'concrete_crack_classification':
                    trainset_origin = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/concrete_crack_classification/train',
                        transform=val_transforms)

                    testset = ImageFolder(
                        root='/home/access/thesis/anomaly_detection/data/additional_datasets/concrete_crack_classification/test',
                        transform=val_transforms)




                indices = [i for i, val in enumerate(trainset_origin.targets) if val not in ANOMALY_CLASSES]
                if few_shot_n_samples>0:
                    with open(
                            '/media/access/a2c3c265-1572-48ed-aacc-59b04b4fd768/thesis/code/PyTorch-Pretrained-ViT/10_shot_learning/inds_dict.pkl',
                            'rb') as handle:
                        inds_dict = pickle.load(handle)
                    indices = np.array(inds_dict[dataset]['1_vs_rest' if one_vs_rest else 'rest_vs_1'][_class])
                    indices = [i for i,flag in enumerate(indices) if flag]


                print_and_add_to_log(f"len of train dataset {len(indices)}", logging)
                trainset = torch.utils.data.Subset(trainset_origin, indices)

                print_and_add_to_log(f"Train dataset len: {len(trainset)}", logging)
                print_and_add_to_log(f"Test dataset len: {len(testset)}", logging)

                # Create datasetLoaders from trainset and testset
                trainsetLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
                testsetLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

                anomaly_targets = [1 if i in ANOMALY_CLASSES else 0 for i in testset.targets]

                extracted_features_path = join(base_feature_path, 'extracted_features')
                logging.info(f"Extracted features")
                if not os.path.exists(extracted_features_path):
                    os.mkdir(extracted_features_path)

                if calculate_features or not os.path.exists(
                        join(extracted_features_path, 'train_pretrained_ViT_features.npy')):
                    if output_train_features:
                        train_features = get_features(model=model, dataLoader=trainsetLoader,
                                                      is_fmnist = dataset=='fmnist')
                        # with open(join(extracted_features_path, f'train_pretrained_ViT_features.npy'), 'rb') as f:
                        #     train_features = np.load(f)

                        with open(join(extracted_features_path, f'train_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, train_features)

                    if output_test_features:
                        test_features = get_features(model=model, dataLoader=testsetLoader,
                                                     is_fmnist = dataset=='fmnist')
                        with open(join(extracted_features_path, f'test_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, test_features)

                else:
                    if output_train_features:
                        print_and_add_to_log(f"loading feature from {extracted_features_path}", logging)
                        with open(join(extracted_features_path, f'train_pretrained_ViT_features.npy'), 'rb') as f:
                            train_features = np.load(f)
                    if output_test_features:
                        with open(join(extracted_features_path, f'test_pretrained_ViT_features.npy'), 'rb') as f:
                            test_features = np.load(f)

                if output_train_features and output_test_features:
                    print_and_add_to_log(f"Calculate KNN score", logging)
                    distances = knn_score(train_features, test_features, n_neighbours=2)
                    auc = roc_auc_score(anomaly_targets, distances)
                    print_and_add_to_log(auc, logging)






def ls_all_features(input_path,
                    output_path,
                    file_name,
                    datasets,
                    manual_class_num_range = None,
                    input_dataset  = None,
                    input_one_vs_rest  = None,
                    one_vs_rest_vals = [True, False],
                    output_one_vs_rest_vals = None ):
    if output_one_vs_rest_vals is None:
        output_one_vs_rest_vals = one_vs_rest_vals

    for dataset in datasets:
        if dataset == 'cifar100':
            _class_num = 20

        elif dataset == 'cats_vs_dogs':
            _class_num = 2

        elif dataset == 'dior':
            _class_num = 19

        else:
            _class_num = 10

        if manual_class_num_range is not None:
            _classes = range(*manual_class_num_range)

        else:
            _classes = range(_class_num)

        for _class in _classes:
            for one_vs_rest, output_one_vs_rest in zip(one_vs_rest_vals , output_one_vs_rest_vals):
                if input_dataset is not None:
                    _input_dataset = input_dataset
                else:
                    _input_dataset = dataset

                if dataset!='cats_vs_dogs':

                    if one_vs_rest:
                        input_base_feature_path = join(input_path, f'1_vs_rest/{_input_dataset}/class_{str(_class)}')
                        if input_one_vs_rest is not None:
                            input_base_feature_path = join(input_path,
                                                           f'{input_one_vs_rest}/{_input_dataset}/class_{str(_class)}')
                    else:
                        input_base_feature_path = join(input_path, f'rest_vs_1/{_input_dataset}/class_{str(_class)}')
                        if input_one_vs_rest is not None:
                            input_base_feature_path = join(input_path, f'{input_one_vs_rest}/{_input_dataset}/class_{str(_class)}')

                    if output_one_vs_rest:
                        output_base_feature_path = join(output_path, f'1_vs_rest/{dataset}/class_{str(_class)}')
                    else:
                        output_base_feature_path = join(output_path, f'rest_vs_1/{dataset}/class_{str(_class)}')

                else:
                    if one_vs_rest:
                        input_base_feature_path = join(input_path, f'1_vs_rest/catsVsDogs/class_{str(_class)}')
                        if input_one_vs_rest is not None:
                            input_base_feature_path = join(input_path, f'{input_one_vs_rest}/catsVsDogs/class_{str(_class)}')
                    else:
                        input_base_feature_path = join(input_path, f'rest_vs_1/catsVsDogs/class_{str(_class)}')
                        if input_one_vs_rest is not None:
                            input_base_feature_path = join(input_path, f'{input_one_vs_rest}/catsVsDogs/class_{str(_class)}')

                    if output_one_vs_rest:
                        output_base_feature_path = join(output_path, f'1_vs_rest/catsVsDogs/class_{str(_class)}')
                    else:
                        output_base_feature_path = join(input_path, f'rest_vs_1/catsVsDogs/class_{str(_class)}')

                if not os.path.exists(output_base_feature_path):
                    os.makedirs(output_base_feature_path)

                file_to_sl = join(input_base_feature_path,'extracted_features', file_name)
                if not os.path.exists(join(output_base_feature_path, 'extracted_features')):
                    os.makedirs(join(output_base_feature_path, 'extracted_features'))
                file_output_path = join(output_base_feature_path, 'extracted_features', file_name)
                assert os.path.exists(file_to_sl), f"file not exists: {file_to_sl}"
                os.symlink(file_to_sl, file_output_path)

def ls_testset_from_class_0(input_path,
                    file_name,
                    datasets,
                    manual_class_num_range = None,
                    input_dataset  = None,
                    one_vs_rest_vals = [True, False]
                            ):


    for dataset in datasets:
        if dataset == 'cifar100':
            _class_num = 20

        elif dataset == 'cats_vs_dogs':
            _class_num = 2
        elif dataset == 'dior':
            _class_num = 19
        elif dataset == 'wbc':
            _class_num = 4
        elif dataset == 'flowers':
            _class_num = 20

        # aditional dataset
        elif dataset == 'blood_cells':
            _class_num = 4
        elif dataset == 'concrete_crack_classification':
            _class_num = 2
        elif dataset == 'covid':
            _class_num = 2
        elif dataset == 'intel_classification':
            _class_num = 6
        elif dataset == 'weather_recognition':
            _class_num = 4
        else:
            _class_num = 10

        if manual_class_num_range is not None:
            _classes = range(*manual_class_num_range)

        else:
            _classes = range(1,_class_num)


        for _class in _classes:
            for one_vs_rest in one_vs_rest_vals:
                if input_dataset is not None:
                    _input_dataset = input_dataset
                else:
                    _input_dataset = dataset

                if dataset!='cats_vs_dogs':
                    if one_vs_rest:
                        input_base_feature_path = join(input_path, f'1_vs_rest/{_input_dataset}/class_0')
                        output_base_feature_path = join(input_path, f'1_vs_rest/{_input_dataset}/class_{str(_class)}')
                    else:
                        input_base_feature_path = join(input_path, f'rest_vs_1/{_input_dataset}/class_0')
                        output_base_feature_path = join(input_path, f'rest_vs_1/{_input_dataset}/class_{str(_class)}')

                else:
                    if one_vs_rest:
                        input_base_feature_path = join(input_path, f'1_vs_rest/catsVsDogs/class_0')
                        output_base_feature_path = join(input_path, f'1_vs_rest/catsVsDogs/class_{str(_class)}')
                    else:
                        input_base_feature_path = join(input_path, f'rest_vs_1/catsVsDogs/class_0')
                        output_base_feature_path = join(input_path, f'rest_vs_1/catsVsDogs/class_{str(_class)}')


                if not os.path.exists(output_base_feature_path):
                    os.makedirs(output_base_feature_path)

                file_to_sl = join(input_base_feature_path,'extracted_features', file_name)
                if not os.path.exists(join(output_base_feature_path, 'extracted_features')):
                    os.makedirs(join(output_base_feature_path, 'extracted_features'))
                file_output_path = join(output_base_feature_path, 'extracted_features', file_name)
                assert os.path.exists(file_to_sl), f"file not exists: {file_to_sl}"
                os.symlink(file_to_sl, file_output_path)

def get_val_transformations(crop_dim=32
                            ):
    return transforms.Compose([
        transforms.CenterCrop(crop_dim),
        transforms.ToTensor(),
    ])



def freeze_model(model):
    # for param in model.parameters():
    #     param.requires_grad = False
    non_freezed_layer = []
    for name, param in model.named_parameters():
        if not (name.startswith('transformer.cloned_block') or name.startswith('cloned_')):
            param.requires_grad = False
        else:
            non_freezed_layer.append(name)
    print("=========================================")
    print("Clone block didn't freezed")
    print(f"layers name: {non_freezed_layer}")
    print("=========================================")
    return

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


def get_simclr_augmentation(resize_factor=0.5, image_size=224):
    # parameter for resizecrop
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    # if P.resize_fix: # if resize_fix is True, use same scale
    #     resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )

    return transform


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def forward_one_epoch(args,
                      loader,
                      optimizer,
                      criterion,
                      net,
                      mode,
                      progress_bar_str,
                      num_of_epochs,
                      device='cuda'
                      ):
    losses, cur_accuracies = [], []
    all_preds, all_targets = [], []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

        if mode == Mode.training:
            optimizer.zero_grad()

        inputs = inputs.to(device)

        origin_block_outputs, cloned_block_outputs = net(inputs)
        loss = criterion(cloned_block_outputs, origin_block_outputs)
        losses.append(loss.item())

        if mode == Mode.training:
            # do a step
            loss.backward()
            optimizer.step()

        if batch_idx % 20 == 0:
            progress_bar(batch_idx, len(loader), progress_bar_str
                         % (num_of_epochs, np.mean(losses), losses[-1]))

        # targets_cpu = origin_block_outputs.detach().cpu().data.numpy()
        # outputs_cpu = [i.detach().cpu().data.numpy() for i in cloned_block_outputs]
        #
        # all_targets.extend(targets_cpu)
        # all_preds.extend(outputs_cpu)

        del inputs, origin_block_outputs, cloned_block_outputs, loss
        torch.cuda.empty_cache()

        if batch_idx>10:
            break
    return losses, all_targets, all_preds


import gc


def train(model, best_model, args, dataloaders,
          model_checkpoint_path,
          output_path, device='cuda',
          seed=42, anomaly_classes = None):


    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    best_model = best_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.MSELoss()

    training_losses, val_losses = [], []

    training_loader = dataloaders['training']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    best_val_loss = np.inf
    no_imporvement_epochs = 0


    # start training
    for epoch in range(1, args['epochs'] + 1):

        # training
        model = model.train()

        progress_bar_str = 'Teain: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

        # if epoch == 1:
        #     print("calculate the initialized train recon scores for notmalization")
        #     train_outputs_recon_scores = get_finetuned_features(args,
        #                                                         model,
        #                                                         training_loader)
        #
        #     if not os.path.exists(join(args['base_feature_path'], 'features_distances')):
        #         os.makedirs(join(args['base_feature_path'], 'features_distances'))
        #     np.save(join(args['base_feature_path'], 'features_distances',
        #                  f'untrained_train_outputs_recon_scores.npy'), train_outputs_recon_scores)
        #
        #     del train_outputs_recon_scores
        #     gc.collect()
        #     model = model.train()
        #     print('=========================================')

        losses, _, _ = forward_one_epoch(
            args = args,
            loader=training_loader,
            optimizer=optimizer,
            criterion=criterion,
            net=model,
            mode=Mode.training,
            progress_bar_str=progress_bar_str,
            num_of_epochs=epoch)

        # save first batch loss for normalization
        train_epoch_loss = np.mean(losses)
        sys.stdout.flush()
        print()
        print(f'Train epoch {epoch}: loss {train_epoch_loss}', flush=True)
        training_losses.append(train_epoch_loss)

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), model_checkpoint_path)

        if epoch==1 or epoch==5:
            init_model_checkpoint_path = join(output_path, f'{epoch}_full_recon_model_state_dict.pkl')
            torch.save(model.state_dict(), init_model_checkpoint_path)

        del losses
        gc.collect()

        if (epoch - 1) % args['eval_every'] == 0:
            # validation
            model.eval()
            progress_bar_str = 'Validation: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

            losses, _, _ = forward_one_epoch(args = args,
                                             loader=val_loader,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             net=model,
                                             mode=Mode.validation,
                                             progress_bar_str=progress_bar_str,
                                             num_of_epochs=epoch
                                             )

            val_epoch_loss = np.mean(losses)
            sys.stdout.flush()

            print()
            print(f'Validation epoch {epoch // args["eval_every"]}: loss {val_epoch_loss}',
                  flush=True)
            val_losses.append(val_epoch_loss)

            #
            cur_acc_loss = {
                'training_losses': training_losses,
                'val_losses': val_losses
            }

            if best_val_loss - 0.001 > val_epoch_loss:
                best_val_loss = val_epoch_loss
                best_acc_epoch = epoch

                print(f'========== new best model! epoch {best_acc_epoch}, loss {best_val_loss}  ==========')

                best_model.load_state_dict(model.state_dict())
                # best_model = copy.deepcopy(model)
                no_imporvement_epochs = 0
            else:
                no_imporvement_epochs += 1

            # del losses
            # gc.collect()

            progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'
            model.eval()
            test_losses, _, _ = forward_one_epoch(
                args=args,
                loader=test_loader,
                optimizer=None,
                criterion=criterion,
                net=model,
                mode=Mode.test,
                progress_bar_str=progress_bar_str,
                num_of_epochs=0)

            test_epoch_loss = np.mean(test_losses)
            print("===================== OOD val Results =====================")
            print(f'OOD val Loss : {test_epoch_loss}')
            del test_losses
            gc.collect()
            if no_imporvement_epochs > args['early_stopping_n_epochs']:
                print(f"Stop due to early stopping after {no_imporvement_epochs} epochs without improvment")
                print(f"epoch number {epoch}")
                break

            if args['plot_every_layer_summarization']:
                _, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                  one_vs_rest=args['one_vs_rest'],
                                                  _class=args['_class'],
                                                  normal_test_sample_only=False,
                                                  use_imagenet=args['use_imagenet']
                                                                )

                eval_test_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
                anomaly_targets = [0 if i in anomaly_classes else 1 for i in testset.targets]

                model = model.eval()
                outputs_recon_scores = get_finetuned_features(args,
                                                              model,
                                                              eval_test_loader)
                outputs_recon_scores = outputs_recon_scores[0]

                print("========================================================")
                for j in range(len(args['use_layer_outputs'])):
                    layer_ind = args['use_layer_outputs'][j]
                    print(f"Layer number: {layer_ind}")
                    print(
                        f"Test Max layer outputs score: {np.max(np.abs(outputs_recon_scores[:, layer_ind]))}")
                    rot_auc = roc_auc_score(anomaly_targets,
                                            outputs_recon_scores[:, layer_ind] )
                    print(f'layer AUROC score: {rot_auc}')
                    print("--------------------------------------------------------")
            model = model.train()

    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

    model = model.eval()
    test_losses, _, _ = forward_one_epoch(args = args,
                                          loader=test_loader,
                                          optimizer=None,
                                          criterion=criterion,
                                          net=model,
                                          mode=Mode.test,
                                          progress_bar_str=progress_bar_str,
                                          num_of_epochs=0)

    best_model = best_model.to('cpu')
    model = model.to('cpu')
    test_epoch_loss = np.mean(test_losses)
    print("===================== OOD val Results =====================")
    print(f'OOD val Loss : {test_epoch_loss}')
    return model, best_model, cur_acc_loss


from torchvision.transforms import *
def get_my_augmentations(is_fmnist):
    if is_fmnist:
        transforms = Compose([
            ToPILImage(),
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ToTensor(),
            # Normalize(**_normalize),
            # ToTensor()
        ]
        )

    else:
        transforms = Compose([
            ToPILImage(),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ToTensor(),
            # Normalize(**_normalize),
            # ToTensor()
        ]
        )

    return transforms


def get_finetuned_features(args,
                               model,
                               loader,
                               ):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = model.to('cuda')
    criterion = torch.nn.MSELoss(reduce=False)

    # start eval
    model = model.eval()
    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f'

    all_outputs_recon_scores = []

    with torch.no_grad():
        outputs_recon_scores = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):


            inputs = inputs.to('cuda')

            origin_block_outputs, cloned_block_outputs = model(inputs)
            # origin_block_outputs, cloned_block_outputs = origin_block_outputs.permute( 1,0,2,3 ), cloned_block_outputs.permute( 1,0,2,3 )
            loss = criterion(cloned_block_outputs, origin_block_outputs)
            loss = torch.mean(loss, [2, 3])
            loss = loss.permute(1, 0)
            outputs_recon_scores.extend(-1 * loss.detach().cpu().data.numpy())

            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(loader), progress_bar_str
                             % (1, np.mean(outputs_recon_scores)))

            del inputs, origin_block_outputs, cloned_block_outputs, loss
            torch.cuda.empty_cache()
        all_outputs_recon_scores.append(outputs_recon_scores)


    return np.array(all_outputs_recon_scores)

def get_transforms(dataset, use_imagenet):

    # 0.5 normalization
    if dataset == 'fmnist':
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]


    else:
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]

    val_transforms = Compose(val_transforms_list)
    return val_transforms

def get_datasets_for_ViT(dataset, one_vs_rest, _class,
                         normal_test_sample_only=True,
                         use_imagenet = False):
    if dataset == 'cifar10':
        number_of_classes = 10

    elif dataset == 'cifar100':
        number_of_classes = 20

    elif dataset == 'fmnist':
        number_of_classes = 10

    elif dataset == 'cats_vs_dogs':
        number_of_classes = 2

    elif dataset == 'dior':
        number_of_classes = 19

    elif dataset == 'wbc':
        number_of_classes = 4

    elif dataset == 'flowers':
        number_of_classes = 102

    # additional datasets
    elif dataset == 'blood_cells':
        number_of_classes = 4

    elif dataset == 'covid':
        number_of_classes = 2

    elif dataset == 'intel_classification':
        number_of_classes = 6

    elif dataset == 'weather_recognition':
        number_of_classes = 4

    elif dataset == 'concrete_crack_classification':
        number_of_classes = 2

    else:
        raise ValueError(f"{dataset} not supported yet!")

    if one_vs_rest:
        ANOMALY_CLASSES = [i for i in range(number_of_classes) if i != _class]
    else:
        ANOMALY_CLASSES = [_class]

    val_transforms = get_transforms(dataset = dataset,
                                    use_imagenet= use_imagenet)

    # get dataset
    if dataset == 'cifar10':
        trainset_origin = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                                  train=True, download=True,
                                  transform=val_transforms)


        # TODO: FIX IT!
        testset = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                          train=False, download=True,
                          transform=val_transforms)

    elif dataset == 'cifar100':
        trainset_origin = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                                   train=True, download=True,
                                   transform=val_transforms)

        testset = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                           train=False, download=True,
                           transform=val_transforms)

        trainset_origin.targets = sparse2coarse(trainset_origin.targets)
        testset.targets = sparse2coarse(testset.targets)


    elif dataset == 'fmnist':
        trainset_origin = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                                       train=True, download=True,
                                       transform=val_transforms)

        testset = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                               train=False, download=True,
                               transform=val_transforms)



    elif dataset == 'cats_vs_dogs':
        trainset_origin = ImageFolder(root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/train',
                                      transform=val_transforms)
        testset = ImageFolder(root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/test',
                              transform=val_transforms)

    elif dataset == 'dior':
        trainset_origin = DiorDataset(
            image_path='/home/access/thesis/anomaly_detection/data/dior/cropped_train_val_images/',
            labels_dict_path='/home/access/thesis/anomaly_detection/data/dior/labels_dict.pkl',
            transform=val_transforms)

        testset = DiorDataset(
            image_path='/home/access/thesis/anomaly_detection/data/dior/cropped_test_images/',
            labels_dict_path='/home/access/thesis/anomaly_detection/data/dior/labels_dict.pkl',
            transform=val_transforms)

    elif dataset == 'wbc':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/wbc/training',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/wbc/test',
            transform=val_transforms)

    elif dataset == 'flowers':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/102flowers/training',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/102flowers/test',
            transform=val_transforms)


    # additional datasets
    elif dataset == 'blood_cells':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/blood_cells/train',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/blood_cells/test',
            transform=val_transforms)

    elif dataset == 'covid':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/covid/train',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/covid/test',
            transform=val_transforms)

    elif dataset == 'intel_classification':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/intel_classification/train',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/intel_classification/test',
            transform=val_transforms)

    elif dataset == 'weather_recognition':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/weather_recognition/train',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/weather_recognition/test',
            transform=val_transforms)

    elif dataset == 'concrete_crack_classification':
        trainset_origin = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/concrete_crack_classification/train',
            transform=val_transforms)

        testset = ImageFolder(
            root='/home/access/thesis/anomaly_detection/data/additional_datasets/concrete_crack_classification/test',
            transform=val_transforms)


    train_indices = [i for i, val in enumerate(trainset_origin.targets) if val not in ANOMALY_CLASSES]
    logging.info(f"len of train dataset {len(train_indices)}")
    trainset = torch.utils.data.Subset(trainset_origin, train_indices)

    if normal_test_sample_only:
        test_indices = [i for i, val in enumerate(testset.targets) if val not in ANOMALY_CLASSES]
        testset = torch.utils.data.Subset(testset, test_indices)

    logging.info(f"len of test dataset {len(testset)}")
    return trainset, testset




def plot_dataset(dataset, number_of_samples, classes=[]):
    assert len(dataset) >= number_of_samples

    # settings
    nrows, ncols = int(np.ceil(number_of_samples / 3)), 3  # array of sub-plots
    figsize = [8, 8]  # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        sample, label = dataset[i - 1]
        sample = sample.permute(1, 2, 0)
        if len(classes) > 0:
            label = classes[label]
        axi.imshow(sample.squeeze(), cmap='gray', vmin=0, vmax=255)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(label)

    plt.tight_layout(True)
    plt.show()



def print_and_add_to_log(msg, logging):
    print(msg)
    logging.info(msg)
def get_val_transformations(mean, std,
                            crop_dim=32,
                            ):
    return transforms.Compose([
        transforms.CenterCrop(crop_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])


def get_datasets(dataset, train_transforms, val_transforms, use_coarse_labels=False,
                 anomaly_classes=None):
    if dataset == 'cifar100':
        testset = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                           train=False, download=True,
                           transform=val_transforms)

        trainset = CIFAR100(root='/home/access/thesis/anomaly_detection/data',
                            train=True, download=True,
                            transform=train_transforms)

        if use_coarse_labels:
            trainset.targets = sparse2coarse(trainset.targets)
            testset.targets = sparse2coarse(testset.targets)

    elif dataset == 'cifar10':
        testset = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                          train=False, download=True,
                          transform=val_transforms)

        trainset = CIFAR10(root='/home/access/thesis/anomaly_detection/data',
                           train=True, download=True,
                           transform=train_transforms)


    elif dataset == 'fmnist':
        trainset = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                                       train=True, download=True,
                                       transform=val_transforms)

        testset = FashionMNIST(root='/home/access/thesis/anomaly_detection/data',
                               train=False, download=True,
                               transform=val_transforms)

    elif dataset == 'cats_vs_dogs':
        trainset= ImageFolder(root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/train',
                                      transform=val_transforms)
        testset = ImageFolder(root='/home/access/thesis/anomaly_detection/data/dogs-vs-cats/dataset/test',
                              transform=val_transforms)

    if anomaly_classes is not None:
        y_train = trainset.targets
        indices = [i for i, val in enumerate(y_train) if val not in anomaly_classes]
        trainset = torch.utils.data.Subset(trainset, indices)

    return trainset, testset


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def plot_tensors_as_images(tensors, number_of_samples):
    assert len(tensors) >= number_of_samples

    # settings
    nrows, ncols = int(np.ceil(number_of_samples / 3)), 3  # array of sub-plots
    figsize = [8, 8]  # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        sample = tensors[i]
        sample = sample.permute(1, 2, 0)
        axi.imshow(sample, cmap='gray', vmin=0, vmax=255)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols

    plt.tight_layout(True)
    plt.show()


class Mode(Enum):
    training = 1
    validation = 2
    test = 3


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_graphs(train_accuracies, val_accuracies, train_losses, val_losses, path_to_save=''):
    plot_accuracy(train_accuracies, val_accuracies, path_to_save=path_to_save)
    plot_loss(train_losses, val_losses, path_to_save=path_to_save)
    return max(val_accuracies)


def plot_accuracy(train_accuracies, val_accuracies, to_show=True, label='accuracy', path_to_save=''):
    print(f'Best val accuracy was {max(val_accuracies)}, at epoch {np.argmax(val_accuracies)}')
    train_len = len(np.array(train_accuracies))
    val_len = len(np.array(val_accuracies))

    xs_train = list(range(0, train_len))

    if train_len != val_len:
        xs_val = list(range(0, train_len, math.ceil(train_len / val_len)))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_accuracies), label='val ' + label)
    plt.plot(xs_train, np.array(train_accuracies), label='train ' + label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/accuracy_graph.png')

    if to_show:
        plt.show()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def plot_loss(train_losses, val_losses, to_show=True, val_label='val loss', train_label='train loss',
              path_to_save=''):
    train_len = len(np.array(train_losses))
    val_len = len(np.array(val_losses))

    xs_train = list(range(0, train_len))
    if train_len != val_len:
        xs_val = list(range(0, train_len, int(train_len / val_len) + 1))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_losses), label=val_label)
    plt.plot(xs_train, np.array(train_losses), label=train_label)

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/loss_graph.png')
    if to_show:
        plt.show()