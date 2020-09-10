import numpy as np
import random
from PIL import Image
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import os


class GroupGenerator(Sequence):
    def __init__(self, image_file_groups, score_groups, saliency_folder, batch_size=16, image_aug=True, shuffle=True, imagenet_pretrain=False):
        self.image_file_groups = image_file_groups
        self.score_groups = score_groups
        self.saliency_folder = saliency_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.imagenet_pretrain = imagenet_pretrain
        if image_aug:
            self.seq = iaa.Sequential([iaa.Fliplr(0.5)])
        self.image_aug = image_aug
        self.on_epoch_end()

    def __len__(self):
        return sum(self.group_length)

    def on_epoch_end(self):
        if self.shuffle:
            images_scores = list(zip(self.image_file_groups, self.score_groups))
            random.shuffle(images_scores)
            self.image_file_groups, self.score_groups = zip(*images_scores)

            self.index_groups = []
            self.group_length = []
            for i in range(len(self.image_file_groups)):
                self.index_groups.append(np.arange(len(self.image_file_groups[i])))
                self.group_length.append(len(self.image_file_groups[i]) // self.batch_size)

            for i in range(len(self.index_groups)):
                np.random.shuffle(self.index_groups[i])

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
        saliency_maps = []
        y_scores = []

        for idx_1 in self.index_groups[idx_0][item * self.batch_size: (item + 1) * self.batch_size]:
            image = np.asarray(Image.open(self.image_file_groups[idx_0][idx_1]), dtype=np.float32)
            saliency_file = os.path.join(self.saliency_folder, os.path.basename(self.image_file_groups[idx_0][idx_1]))
            saliency_image = Image.open(saliency_file)
            saliency_map = saliency_image.resize((int(image.shape[1] / 32), int(image.shape[0] / 32)))
            saliency_map = np.asarray(saliency_map, dtype=np.float32)
            saliency_map /= 255.
            if self.imagenet_pretrain:
                image /= 127.5
                image -= 1.
            else:
                image[:, :, 0] -= 117.27205081970828
                image[:, :, 1] -= 106.23294835284031
                image[:, :, 2] -= 94.40750328714887
                image[:, :, 0] /= 59.112836751661085
                image[:, :, 1] /= 55.65498543815568
                image[:, :, 2] /= 54.9486100975773
            images.append(image)
            saliency_maps.append(saliency_map)
            y_scores.append(self.score_groups[idx_0][idx_1])

        if self.image_aug:
            images_aug = self.seq(images=images)
            saliencys_aug = self.seq(images=saliency_maps)
            return [np.array(images_aug), np.array(saliencys_aug)], np.array(y_scores)
        else:
            return [np.array(images), np.array(saliency_maps)], np.array(y_scores)


