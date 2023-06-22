# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from utils import *
from models import *
from data_transform import *

from pathlib import Path

DATASET_LOC = "./dataset"
DATA_LOC = "./data"

"""
Downloads and prepares the dataset for knowledge distillation
Basically, it just runs the model on the dataset and saves the output for later compression.
(To easily calculate the similarity between the outputs of the model for different inputs)
"""
def prepare_dataset(model, dataset_name, overwrite=False):
    datasets = ["cifar10", "cifar100", "lfw", "gtsrb", "celeba"]
    dataset_name = dataset_name.lower()

    out_name = DATASET_LOC + "/distill_" + dataset_name

    if os.path.exists(out_name) and not overwrite:
        print(f"Dataset {dataset_name} already exists, skipping...")
        return

    if dataset_name not in datasets:
        raise ValueError("Dataset not supported, please choose from " + str(datasets))

    if dataset_name == "cifar100":
        test_dataset = torchvision.datasets.CIFAR100(
            root=DATA_LOC, train=True, download=True, transform=cifar100_transforms
        )
    elif dataset_name == "lfw":
        test_dataset = torchvision.datasets.LFWPeople(
            root=DATA_LOC, train=True, download=True, transform=LFW_transforms
        )
    elif dataset_name == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_LOC, train=True, download=True, transform=cifar10_transforms_train
        )
    elif dataset_name == "gtsrb":
        test_dataset = torchvision.datasets.GTSRB(
            root=DATA_LOC, train=True, download=True, transform=gtsrb_transforms_train
        )
    elif dataset_name == "celeba":
        test_dataset = torchvision.datasets.CelebA(
            root=DATA_LOC, split="train", download=True, transform=VGGFace_transforms
        )

    unloader = transforms.ToPILImage()

    p = Path(DATASET_LOC)
    p.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(test_dataset, batch_size=1)

    device = next(model.parameters()).device

    model.eval()
    list_clean_data_knowledge_distill = []
    for i, (input, target) in enumerate(dataloader):
        input, target = input.to(device), target.to(device)

        with torch.no_grad():
            output = model(input)

        input = input.squeeze(0)
        input = unloader(input)
        output = output.squeeze(0)
        list_clean_data_knowledge_distill.append((input, output))

    torch.save(list_clean_data_knowledge_distill, out_name)
