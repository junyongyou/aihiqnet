import numpy as np
import os
import time
import glob
import random
from PIL import Image
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from pickle import dump


class ResolutionGroupGenerator(Sequence):
    """
    Generator to supply group image data, individual dataset should go to individual group because they can have different resolutions
    """
    def __init__(self, images_scores, batch_size=16, image_aug=False, shuffle=True, imagenet_pretrain=True):
        self.images_scores = images_scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.imagenet_pretrain = imagenet_pretrain
        if image_aug:
            # do image augmentation by left-right flip
            self.seq = iaa.Sequential([iaa.Fliplr(0.5)])
        self.image_aug = image_aug
        self.generate_groups()
        self.on_epoch_end()

    def __len__(self):
        return sum(self.group_length)

    def generate_groups(self):
        """
        Group images based on their resolutions
        :return:
        """
        resolution_group = dict()
        for image_score in self.images_scores:
            content = image_score.split(';')
            image = Image.open(content[0])
            # score = content[1:]
            image_size = '{}_{}'.format(image.size[0], image.size[1])
            if image_size in resolution_group:
                images_scores_per_resolution = resolution_group.get(image_size)
                images_scores_per_resolution.append(image_score)
                resolution_group.update({image_size: images_scores_per_resolution})
            else:
                images_scores_per_resolution = [image_score]
                resolution_group[image_size] = images_scores_per_resolution

        for resolution in resolution_group:
            images_scores_per_resolution = resolution_group.get(resolution)
            len_images_scores = len(images_scores_per_resolution)
            if len_images_scores < self.batch_size:
                gap = self.batch_size - len_images_scores
                random_items = random.sample(images_scores_per_resolution, gap)
                images_scores_per_resolution.extend(random_items)
                resolution_group.update({resolution: images_scores_per_resolution})
            else:
                mod = len_images_scores % self.batch_size
                if mod >= self.batch_size / 2:
                    random_items = random.sample(images_scores_per_resolution, mod)
                    images_scores_per_resolution.extend(random_items)
                    resolution_group.update({resolution: images_scores_per_resolution})

        self.resolution_groups = resolution_group

    def on_epoch_end(self):
        if self.shuffle:
            self.index_groups = []
            self.resolution_index = []
            resolutions = list(self.resolution_groups.keys())
            random.shuffle(resolutions)
            self.group_length = []
            for i, resolution in enumerate(resolutions):
                images_scores_per_resolution = self.resolution_groups.get(resolution)
                random.shuffle(images_scores_per_resolution)
                self.index_groups.append(np.arange(len(images_scores_per_resolution)))
                self.resolution_index.append(resolution)
                self.group_length.append(len(images_scores_per_resolution) // self.batch_size)

    def __getitem__(self, item):
        lens = 0
        idx_0 = len(self.group_length) - 1
        for i, data_len in enumerate(self.group_length):
            lens += data_len
            if item < lens:
                idx_0 = i
                break
        item -= (lens - self.group_length[idx_0])

        images = []
        y_scores = []

        resolution = self.resolution_index[idx_0]
        images_scores = self.resolution_groups.get(resolution)
        for idx_1 in range(item * self.batch_size, (item + 1) * self.batch_size):
            image_score = images_scores[idx_1].split(';')
            image = np.asarray(Image.open(image_score[0]), dtype=np.float32)
            if self.imagenet_pretrain:
                # ImageNet normalization
                image /= 127.5
                image -= 1.
            else:
                # Normalization based on the combined database consisting of KonIQ-10k and LIVE-Wild datasets
                image[:, :, 0] -= 117.27205081970828
                image[:, :, 1] -= 106.23294835284031
                image[:, :, 2] -= 94.40750328714887
                image[:, :, 0] /= 59.112836751661085
                image[:, :, 1] /= 55.65498543815568
                image[:, :, 2] /= 54.9486100975773
            images.append(image)

            if ',' in image_score[1]:
                split_score = image_score[1].replace('[', '').replace(']', '').split(',')
                score = [float(s) for s in split_score]
            else:
                score = float(image_score[1])
            y_scores.append(score)

        if self.image_aug:
            images_aug = self.seq(images=images)
            return np.array(images_aug), np.array(y_scores)
        else:
            return np.array(images), np.array(y_scores)
