import os
from pickle import load
import numpy as np
from PIL import Image
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from image_quality.models.image_quality_model import phiq_net
from image_quality.train.train import get_image_list_path


def evaluation(images_scores, using_single_mos, model, draw_scatter=True):
    mos_scales = np.array([1, 2, 3, 4, 5])
    mos_scores = []
    predictions = []
    for k, image_file_score in enumerate(images_scores):
        content = image_file_score.split(';')
        image_file = content[0]
        if using_single_mos:
            score = float(content[-1])
        else:
            split_score = content[1].replace('[', '').replace(']', '').split(',')
            score_distribution = [float(s) for s in split_score]
            score = np.sum(np.multiply(mos_scales, score_distribution))
        image = Image.open(image_file)
        image = np.asarray(image, dtype=np.float32)
        image /= 127.5
        image -= 1.

        # start_time = time.time()
        if using_single_mos:
            prediction = model.predict(np.expand_dims(image, axis=0))[0][0]
        else:
            prediction = model.predict(np.expand_dims(image, axis=0))
            prediction = np.sum(np.multiply(mos_scales, prediction[0]))

        mos_scores.append(score)

        predictions.append(prediction)
        print('NUM: {}, {}, Real score: {}, predicted: {}'.format(k, image_file, score, prediction))

    PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
    SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
    RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
    MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
    print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))

    if draw_scatter:
        axes = plt.gca()

        axes.set_xlim([1, 5])
        axes.set_ylim([1, 5])
        line = mlines.Line2D([1, 5], [1, 5], color='gray')

        axes.scatter(mos_scores, predictions, color='c', s=10, alpha=0.4, cmap='viridis')
        axes.add_line(line)

        axes.set_xlabel('Normalized MOS')
        axes.set_ylabel('Prediction')


def main():
    args = {}
    args['n_quality_levels'] = 1

    # Choose between 'resnet50', 'densnet121', 'vgg16'
    args['backbone'] = 'resnet50'
    # args['backbone'] = 'densnet121'
    # args['backbone'] = 'vgg16'

    # Choose between False and True, default: False
    args['naive_backbone'] = False

    # Image and score must be provided
    args['images_scores_file'] = r'..\databases\train_val_test_koniq.pkl'
    args['image_folder'] = r'..\databaseskoniq_all'

    args['feature_fusion'] = True
    args['attention_module'] = True

    args['weights'] = r''

    separate_test_file = r'..\databases\train_val_test_koniq.pkl'
    with open(separate_test_file, 'rb') as f:
        separate_train_images_scores, separate_val_images_scores, separate_test_images_scores = load(f)
    separate_images_scores = separate_train_images_scores
    separate_images_scores.extend(separate_val_images_scores)
    separate_images_scores.extend(separate_test_images_scores)

    # separate_images_scores = separate_test_images_scores

    image_folder = r'..\databases\koniq_all'
    separate_images_scores = get_image_list_path(image_folder, separate_images_scores,
                                                 args['n_quality_levels'],
                                                 do_normalization=True)

    model = phiq_net(n_quality_levels=args['n_quality_levels'],
                     naive_backbone=args['naive_backbone'],
                     backbone=args['backbone'],
                     feature_fusion=args['feature_fusion'],
                     attention_module=args['attention_module'])
    model.load_weights(args['weights'])

    using_single_MOS = True
    evaluation(separate_images_scores, using_single_MOS, model)


if __name__ == '__main__':
    main()