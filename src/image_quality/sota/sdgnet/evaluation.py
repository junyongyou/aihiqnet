from tensorflow.keras.callbacks import Callback
import numpy as np
from PIL import Image
import scipy.stats
from sklearn.metrics import mean_squared_error


class ModelEvaluationIQ(Callback):
    def __init__(self, image_files, scores, imagenet_pretrain=False):
        super(ModelEvaluationIQ, self).__init__()
        self.image_files = image_files
        self.scores = scores
        self.imagenet_pretrain = imagenet_pretrain

    def __evaluation_single_mos__(self):
        scores = []
        predictions = []
        for image_file, score in zip(self.image_files, self.scores):
            image = np.asarray(Image.open(image_file)).astype(np.float32)
            if self.imagenet_pretrain:
                image /= 127.5
                image -= 1.
            prediction = self.model.predict(np.expand_dims(image, axis=0))
            scores.append(score)
            predictions.append(prediction[0][0][0])
            # print('{}: prediction - {}, score - {}'.format(image_file, prediction[0][0], score))

        PLCC = scipy.stats.pearsonr(scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(scores, predictions)[0]
        RMSE = np.sqrt(mean_squared_error(predictions, scores))
        # MAD = np.mean(np.abs(np.subtract(predictions, scores)))
        print('PLCC: {}, SRCC: {}, RMSE: {}'.format(PLCC, SRCC, RMSE))
        return PLCC, SRCC, RMSE

    def on_epoch_end(self, epoch, logs=None):
        plcc, srcc, rmse = self.__evaluation_single_mos__()

        logs['plcc'] = plcc
        logs['srcc'] = srcc
        logs['rmse'] = rmse
