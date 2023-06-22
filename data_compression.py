from utils import *
import torch
import numpy as np
import random
from sklearn import preprocessing


def compress_data_train(dataset_name, com_ratio, batch_size=200, overwrite=False):
    dataset_name = dataset_name.lower()
    out_name = "./dataset/compression_" + dataset_name + "_" + str(com_ratio)

    if os.path.exists(out_name) and not overwrite:
        print(f"Reduced dataset {dataset_name}_{com_ratio} already exists, skipping...")
        return

    dataset = torch.load("./dataset/distill_" + dataset_name)

    random.shuffle(dataset)
    data_num = len(dataset)
    print("distill_data num:", data_num)

    images = []
    outputs = []
    for i in range(data_num):
        img = np.array(dataset[i][0]).flatten()
        output = np.array(dataset[i][1].cpu())
        img = img.reshape(1, -1)
        images.append(preprocessing.normalize(img, norm="l2").squeeze())
        output = output.reshape(1, -1)
        outputs.append(preprocessing.normalize(output, norm="l2").squeeze())

    images = np.array(images)
    outputs = np.array(outputs)

    batch_num = int(data_num / batch_size) + (data_num % batch_size != 0)

    data_compression = []

    def select_img(images_batch, outputs_batch, batch_n):
        data_num = images_batch.shape[0]
        max_num = int(data_num * com_ratio)
        if max_num == 0:
            return

        n_selected = 0
        images_sim = np.dot(images_batch, images_batch.transpose())

        outputs_sim = np.dot(outputs_batch, outputs_batch.transpose())
        co_sim = np.multiply(images_sim, outputs_sim)

        index = random.randint(0, data_num - 1)

        while n_selected < max_num:
            index = np.argmin(co_sim[index])
            data_compression.append(dataset[batch_n * batch_size + index])
            n_selected += 1
            co_sim[:, index] = 1

    for i in range(batch_num):
        images_batch = images[i * batch_size : min((i + 1) * batch_size, data_num)]
        outputs_batch = outputs[i * batch_size : min((i + 1) * batch_size, data_num)]

        select_img(images_batch, outputs_batch, i)

    print("compressed_data num:", len(data_compression))

    torch.save(data_compression, out_name)
