from tensorflow.keras.callbacks import Callback
import numpy as np
from PIL import Image
import scipy.stats
import os


class ModelEvaluationIQ(Callback):
    def __init__(self, image_files, scores, using_single_mos, saliency_folder, imagenet_pretrain=False):
        super(ModelEvaluationIQ, self).__init__()
        self.image_files = image_files
        self.scores = scores
        self.using_single_mos = using_single_mos
        self.saliency_folder = saliency_folder
        self.imagenet_pretrain = imagenet_pretrain
        self.mos_scales = np.array([1, 2, 3, 4, 5])
        # self.live_images, self.live_scores = ImageProvider.generate_live_images()

    def __get_prediction_mos(self, image, saliency_map):
        prediction = self.model.predict([np.expand_dims(image, axis=0), np.expand_dims(saliency_map, axis=0)])
        return prediction[0][0]

    def __get_prediction_distribution(self, image, saliency_map):
        # debug_model = Model(inputs=self.model.inputs, outputs=self.model.get_layer('fpn_concatenate').output)
        # debug_results = debug_model.predict(np.expand_dims(image, axis=0))

        prediction = self.model.predict([np.expand_dims(image, axis=0), np.expand_dims(saliency_map, axis=0)])
        prediction = np.sum(np.multiply(self.mos_scales, prediction[0]))
        return prediction

    def __evaluation__(self, image_files, scores, target_width=512, target_height=512, resize=False):
        predictions = []
        mos_scores = []

        for image_file, score in zip(image_files, scores):
            image = Image.open(image_file)
            saliency_file = os.path.join(self.saliency_folder, os.path.basename(image_file))
            saliency_image = Image.open(saliency_file)
            if 'normal' in image_file:
                saliency_map = saliency_image.resize((32, 24))
            else:
                saliency_map = saliency_image.resize((16, 12))
            saliency_map = np.asarray(saliency_map, dtype=np.float32)
            saliency_map /= 255.
            if resize:
                image = np.asarray(image.resize((target_height, target_width)), dtype=np.float32)
            else:
                image = np.asarray(image, dtype=np.float32)
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

            if self.using_single_mos:
                mos_scores.append(score)
                prediction = self.__get_prediction_mos(image, saliency_map)
            else:
                mos_scores.append(np.sum(np.multiply(self.mos_scales, score)))
                prediction = self.__get_prediction_distribution(image, saliency_map)

            predictions.append(prediction)

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))
        return PLCC, SRCC, RMSE

    def on_epoch_end(self, epoch, logs=None):
        plcc, srcc, rmse = self.__evaluation__(self.image_files, self.scores)

        logs['plcc'] = plcc
        logs['srcc'] = srcc
        logs['rmse'] = rmse
