import numpy as np
from soat.sdg_net.utilitiestrain import preprocess_imagesandsaliencyforiqa
from tensorflow.keras.utils import Sequence


class IQGenerator(Sequence):
    def __init__(self, image_files, saliency_image_files, scores, batch_size=1, shuffle=True):
        self.image_files = image_files
        self.saliency_image_files = saliency_image_files
        self.scores = scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.shape_r = 384
        self.shape_c = 512

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, item):
        indices_batch = self.indices[item * self.batch_size: (item + 1) * self.batch_size]
        image_files = []
        saliency_image_files = []
        y_scores = []
        for index in indices_batch:
            image_files.append(self.image_files[index])
            saliency_image_files.append(self.saliency_image_files[index])
            y_scores.append(self.scores[index])

        X, X2 = preprocess_imagesandsaliencyforiqa(image_files, saliency_image_files, self.shape_r, self.shape_c,
                                                   mirror=False, crop_h=self.shape_r, crop_w=self.shape_c)

        return [X], [np.array(y_scores), X2]
