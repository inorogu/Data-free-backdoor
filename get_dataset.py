"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

import cv2
import random
import pickle
import os
import sys
import torch

version = sys.version_info

import numpy as np
import scipy.io as sio
import PIL.Image as Image
from functools import reduce
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from tensors_dataset import TensorDataset
from multiprocessing.dummy import Pool as ThreadPool
from utils import *
import torchvision

configs = read_config()


def load_training(compressed, train_dataset_name, com_ratio):
    if compressed:
        train_dataset = torch.load(
            "./dataset/compression_" + train_dataset_name + "_" + str(com_ratio)
        )
    else:
        train_dataset = torch.load("./dataset/distill_" + train_dataset_name)

    print("distill_data num:", len(train_dataset))
    train_images = []
    train_labels = []
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        label = train_dataset[i][1].cpu()
        train_images.append(img)
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    print("load train data finished")

    print(type(train_images), type(train_images[0]))
    print(type(train_labels), type(train_labels[0]))

    return train_images, train_labels


def load_testing(test_dataset_name):
    datasets = ["cifar10", "cifar100", "lfw", "gtsrb", "celeba"]
    test_dataset_name = test_dataset_name.lower()

    if test_dataset_name not in datasets:
        raise ValueError("Dataset not supported, please choose from " + str(datasets))
    
    if test_dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    elif test_dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    elif test_dataset_name == "lfw":
        dataset = torchvision.datasets.LFWPeople(root="./data", train=False, download=True)
    elif test_dataset_name == "gtsrb":
        dataset = torchvision.datasets.GTSRB(root="./data", train=False, download=True)
    elif test_dataset_name == "celeba":
        dataset = torchvision.datasets.CelebA(root="./data", train=False, download=True)

    test_images = [dataset[i][0] for i in range(len(dataset))]
    test_labels = dataset.targets

    return test_images, test_labels


def get_dataset(filedir, dataset="TODO: refactor", max_num=0):
    label_num = len(os.listdir(filedir))

    namelist = []
    for i in range(label_num):
        namelist.append(str(i).zfill(5))
    print(
        "multi-thread Loading "
        + str(dataset)
        + " dataset, needs more than 10 seconds ..."
    )

    images = []
    labels = []

    def read_images(i):
        if max_num != 0:
            n = 0
        for filename in os.listdir(filedir + namelist[i]):
            labels.append(i)
            images.append(filedir + namelist[i] + "/" + filename)

            if max_num != 0:
                n += 1
                if n == max_num:
                    break

    return load_dataset_shuffled(read_images, label_num)
