#!/usr/bin/python2
import numpy as np
import scipy.stats
from numpy.lib.stride_tricks import as_strided
from image_quality.misc.imageset_handler import get_image_score_from_groups

import chainer
import scipy.stats
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import os
import six
import imageio
import numbers

from nr_model import Model
from image_quality.misc.imageset_handler import get_image_scores, get_image_score_from_groups


def extract_patches(arr, patch_shape=(32, 32, 3), extraction_step=32):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


# def get_image_scores(mos_file):
#     image_files = {}
#     with open(mos_file, 'r+') as f:
#         lines = f.readlines()
#         for line in lines:
#             content = line.split(',')
#             image_file = content[0]
#             score = float(content[1]) / 25. + 1
#             image_files[image_file] = score
#
#     return image_files


# parser = argparse.ArgumentParser(description='evaluate.py')
# parser.add_argument('INPUT', help='path to input image')
# parser.add_argument('REF', default="", nargs="?", help='path to reference image, if omitted NR IQA is assumed')
# parser.add_argument('--model', '-m', default='',
#                     help='path to the trained model')
# parser.add_argument('--top', choices=('patchwise', 'weighted'),
#                     default='weighted', help='top layer and loss definition')
# parser.add_argument('--gpu', '-g', default=0, type=int,
#                     help='GPU ID')
# args = parser.parse_args()

# chainer.global_config.train = False
# chainer.global_config.cudnn_deterministic = True
#
# FR = True
# # if args.REF == "":
# #     FR = False
#
# # if FR:
# #     model = FRModel(top=args.top)
# # else:
# model_tops = ['patchwise', 'weighted']
# data_sets = ['live', 'tid']
#
# results = []
# for data_set in data_sets:
#     for model_top in model_tops:
#         result_file = os.path.join(r'D:\Academic\Submission\TIP2020\Revision\deepIQA_LIVE_{}_{}.csv'.format(data_set, model_top))
#         rf = open(result_file, 'w+')
#         model = Model(top=model_top)
#
#         # cuda.cudnn_enabled = True
#         # cuda.check_cuda_available()
#         # xp = cuda.cupy
#         model_path = r'D:\Downloads\deepIQA-master\models\nr_{}_{}.model'.format(data_set, model_top)
#         serializers.load_hdf5(model_path, model)
#         # model.to_gpu()
#
#         val_folders = [
#             # r'..\databases\val\koniq_normal',]
#             # r'..\databases\val\koniq_small', ]
#                     r'..\databases\train\live',
#                     r'..\databases\val\live']
#
#         koniq_mos_file = r'..\databases\koniq10k_images_scores.csv'
#         live_mos_file = r'..\databases\live_wild\live_mos.csv'
#
#         image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=True)
#         test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)
#
#         test_image_files = []
#         test_scores = []
#         for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
#             test_image_files.extend(test_image_file_group)
#             test_scores.extend(test_score_group)
#
#         iqa_scores = []
#         k = 0
#         for image_file, score in zip(test_image_files, test_scores):
#             img = imageio.imread(image_file)
#             patches = extract_patches(img)
#             X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
#
#             y = []
#             weights = []
#             batchsize = min(2000, X.shape[0])
#             # t = xp.zeros((1, 1), np.float32)
#             t = np.zeros((1, 1), np.float32)
#             for i in six.moves.range(0, X.shape[0], batchsize):
#                 X_batch = X[i:i + batchsize]
#                 # X_batch = xp.array(X_batch.astype(np.float32))
#                 X_batch = np.array(X_batch.astype(np.float32))
#
#                 model.forward(X_batch, t, False, X_batch.shape[0])
#
#                 # y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
#                 y.append(np.asarray(model.y[0].data).reshape((-1,)))
#                 # weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))
#                 weights.append(np.asarray(model.a[0].data).reshape((-1,)))
#
#             y = np.concatenate(y)
#             weights = np.concatenate(weights)
#
#             iqa_score = np.sum(y * weights) / np.sum(weights)
#             iqa_scores.append(iqa_score)
#
#             print('K: {}, Real score: {}, predicted: {}'.format(k, score, iqa_score))
#             k += 1
#             rf.write('{},{},{}\n'.format(image_file, score, iqa_score))
#         rf.flush()
#         rf.close()
#
#         PLCC = scipy.stats.pearsonr(test_scores, iqa_scores)[0]
#         SRCC = scipy.stats.spearmanr(test_scores, iqa_scores)[0]
#         RMSE = np.sqrt(np.mean(np.subtract(iqa_scores, test_scores) ** 2))
#         MAD = np.mean(np.abs(np.subtract(iqa_scores, test_scores)))
#         print('\n{}, {}, PLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(data_set, model_top, PLCC, SRCC, RMSE, MAD))
#
#         results.append([PLCC, SRCC, RMSE])
#
# print(results)


def parse_result_file():
    result_files = [r'D:\Academic\Submission\TIP2020\Revision\deepIQA_LIVE_live_patchwise.csv',
                    r'D:\Academic\Submission\TIP2020\Revision\deepIQA_LIVE_live_weighted.csv',
                    r'D:\Academic\Submission\TIP2020\Revision\deepIQA_LIVE_tid_patchwise.csv',
                    r'D:\Academic\Submission\TIP2020\Revision\deepIQA_LIVE_tid_weighted.csv']
    for result_file in result_files:
        with open(result_file, 'r+') as rf:
            content = rf.readlines()
            contents = content[0].strip().split(r'..\databases')
        scores = []
        predictions = []
        for content in contents:
            if 'live' in content:
                values = content.split(',')
                scores.append(float(values[1]))
                prediction = 1 + ((100 - float(values[-1])) / 25.)
                # if 'deepIQA_live' in result_file:
                #     prediction = 1 + ((100 - float(values[-1])) / 25.)
                # else:
                #     prediction = 1 + (float(values[-1]) * 4. / 9.)
                predictions.append(prediction)
        PLCC = scipy.stats.pearsonr(scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, scores)))
        print('PLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))


if __name__ == '__main__':
    parse_result_file()


