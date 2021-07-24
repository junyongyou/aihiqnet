import os

import tensorflow.keras.backend as K
import numpy as np
from PIL import Image
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time


class ModelEvaluation:
    """
    Evaluation the model, this script is actually a copy of evaluation callback.
    """
    def __init__(self, model, using_single_mos, imagenet_pretrain=False):
        self.model = model
        self.using_single_mos = using_single_mos
        self.imagenet_pretrain = imagenet_pretrain
        self.mos_scales = np.array([1, 2, 3, 4, 5])

    def __get_prediction_mos(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        # prediction = K.get_value(self.model(np.expand_dims(image, axis=0)))
        return prediction[0][0]

    def __get_prediction_distribution(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        prediction = np.sum(np.multiply(self.mos_scales, prediction[0]))
        return prediction

    def __evaluation__(self, weights, images_scores, evaluation_name='test', result_folder=None, draw_scatter=False):
        self.model.load_weights(weights)

        predictions = []
        mos_scores = []
        if result_folder is not None:
            result_file = os.path.join(result_folder, '{}.csv'.format(evaluation_name))
            if os.path.exists(result_file):
                rf = open(result_file, 'a+')
            else:
                rf = open(result_file, 'w+')

        k = 0
        for image_file_score in images_scores:
            content = image_file_score.split(';')
            image_file = content[0]
            if self.using_single_mos:
                score = float(content[-1])
            else:
                split_score = content[1].replace('[', '').replace(']', '').split(',')
                score_distribution = [float(s) for s in split_score]
                score = np.sum(np.multiply(self.mos_scales, score_distribution))
            image = Image.open(image_file)
            image = np.asarray(image, dtype=np.float32)
            if self.imagenet_pretrain: # image normalization using TF approach
                image /= 127.5
                image -= 1.
            else: # Image normalization by subtracting mean and dividing std
                image[:, :, 0] -= 117.27205081970828
                image[:, :, 1] -= 106.23294835284031
                image[:, :, 2] -= 94.40750328714887
                image[:, :, 0] /= 59.112836751661085
                image[:, :, 1] /= 55.65498543815568
                image[:, :, 2] /= 54.9486100975773

            # start_time = time.time()
            if self.using_single_mos:
                prediction = self.__get_prediction_mos(image)
            else:
                prediction = self.__get_prediction_distribution(image)
            k += 1

            mos_scores.append(score)

            predictions.append(prediction)
            print('NUM: {}, Real score: {}, predicted: {}'.format(k, score, prediction))

            if result_folder is not None:
                rf.write('{},{},{}\n'.format(image_file, score, prediction))

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))

        if result_folder is not None:
            rf.write('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))

        if draw_scatter:
            axes = plt.gca()

            axes.set_xlim([1, 5])
            axes.set_ylim([1, 5])
            line = mlines.Line2D([1, 5], [1, 5], color='gray')

            axes.scatter(mos_scores, predictions, color='c', s=10, alpha=0.4, cmap='viridis')
            axes.add_line(line)

            axes.set_xlabel('Normalized MOS')
            axes.set_ylabel('Prediction')

            plt.savefig(os.path.join(result_folder, '{}.png'.format(evaluation_name)))
            # plt.show()

        if result_folder is not None:
            rf.close()
