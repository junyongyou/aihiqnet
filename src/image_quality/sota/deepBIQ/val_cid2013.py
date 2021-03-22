import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
import scipy.io as sio
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from pickle import dump, load
from common_model import *
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



def cal_lcc(x, y):
    n = x.shape[0]
    s1 = n * ((x * y).sum())
    s2 = x.sum() * y.sum()
    s3 = np.sqrt(n * ((x * x).sum()) - np.square(x.sum()))
    s4 = np.sqrt(n * ((y * y).sum()) - np.square(y.sum()))
    lcc = (s1 - s2) / (s3 * s4)
    return lcc


def cal_features(image_files):
    crop_w = 224
    crop_h = 224
    crop_num_w = 5
    crop_num_h = 5
    normalize = get_imagenet_normalize()
    feature_model = FeatureMode(None)

    img_transform = transforms.Compose([transforms.ToTensor(), normalize])
    for i, train_image_file in enumerate(image_files):
        extname = os.path.splitext(os.path.basename(train_image_file))[1]
        img = Image.open(train_image_file)
        crop_imgs = np.array([])
        img_w, img_h = img.size
        crop_box = get_crop_box(img_w, img_h, crop_w, crop_h, crop_num_w, crop_num_h)
        for box in crop_box:
            crop_imgs = np.append(crop_imgs, img_transform(img.crop(box)).numpy())
        crop_imgs = crop_imgs.reshape(crop_num_w * crop_num_h, 3, 224, 224)
        crop_imgs = torch.from_numpy(crop_imgs).float()
        crop_out = feature_model.extract_feature(crop_imgs)
        crop_out = np.average(crop_out, axis=0)
        feature_path = train_image_file.replace(r'..\databases',
                                                r'..\databases\AlexNet_features').replace(
            extname, '.npy')
        feature_folder = os.path.dirname(feature_path)
        if not os.path.exists(feature_folder):
            os.makedirs(feature_folder)
        np.save(feature_path, crop_out)
        print('Num: {}, {} done'.format(i, train_image_file))
        t = 0

def collect_features_scores(image_files, scores):
    X = np.array([])
    Y = np.array([])

    for image_file, score in zip(image_files, scores):
        extname = os.path.splitext(os.path.basename(image_file))[1]
        feature_path = image_file.replace(r'F:\CID2013',
                                                r'..\databases\AlexNet_features\cid2013').replace(
            extname, '.npy')
        feature = np.load(feature_path)
        X = np.append(X, feature)
        Y = np.append(Y, score)
    X = X.reshape(-1, 4096)
    return X, Y


def evaluation(mos_scores, predictions, draw_scatter=False):

    PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
    SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
    RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
    MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
    print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(PLCC, SRCC, RMSE, MAD))
    # print('Num: {}, total_time: {}, avg_time: {}'.format(k, t, t / k))

    if draw_scatter:
        axes = plt.gca()
        # fig, ax = plt.subplots()

        axes.set_xlim([1, 5])
        axes.set_ylim([1, 5])
        line = mlines.Line2D([1, 5], [1, 5], color='gray')

        axes.scatter(mos_scores, predictions, color='c', s=10, alpha=0.4, cmap='viridis')
        axes.add_line(line)

        axes.set_xlabel('Normalized MOS')
        axes.set_ylabel('Prediction')
        plt.show()

    return PLCC, SRCC, RMSE


def get_folder_from_name(image_file):
    if '_I_' in image_file:
        folder_1 = 'IS1'
    if '_II_' in image_file:
        folder_1 = 'IS2'
    if '_III_' in image_file:
        folder_1 = 'IS3'
    if '_IV_' in image_file:
        folder_1 = 'IS4'
    if '_V_' in image_file:
        folder_1 = 'IS5'
    if '_VI_' in image_file:
        folder_1 = 'IS6'
    if '_C01_' in image_file:
        folder_2 = 'co1'
    if '_C02_' in image_file:
        folder_2 = 'co2'
    if '_C03_' in image_file:
        folder_2 = 'co3'
    if '_C04_' in image_file:
        folder_2 = 'co4'
    if '_C05_' in image_file:
        folder_2 = 'co5'
    if '_C06_' in image_file:
        folder_2 = 'co6'
    if '_C07_' in image_file:
        folder_2 = 'co7'
    if '_C08_' in image_file:
        folder_2 = 'co8'
    return folder_1, folder_2

def get_image_scores(image_folder, mos_file):
    image_files = []
    image_scores = []
    with open(mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split()
            folder_1, folder_2 = get_folder_from_name(content[0])
            image_file = os.path.join(image_folder[0], folder_1, folder_2, content[0] + '.jpg')
            score = float(content[3]) / 25. + 1
            image_files.append(image_file)
            image_scores.append(score)

    return image_files, image_scores


def validation():
    val_folders = [r'F:\CID2013']
    # val_folders = [r'F:\SPAG_image_quality_dataset\SPAQ\TestImage']
    cid_mos_file = r'F:\CID2013\image_mos.txt'
    # spaq_mos_file = r'..\databases\spaq\image_mos.csv'
    test_image_files, test_scores = get_image_scores(val_folders, cid_mos_file)

    X_test, y_test = collect_features_scores(test_image_files, test_scores)
    with (open(r'..\databases\results\deepBIQ\scaler_spaq.obj', 'rb')) as scaler_file:
        scaler_x = load(scaler_file)

    # scaler_x = joblib.load(r'..\databases\results\deepBIQ\scaler_koniq_all_1.pkl')
    X_test = scaler_x.transform(X_test)

    # with open(r'..\databases\results\deepBIQ\biq_model_normal.pkl', 'rb') as cf:
    #     clf = load(cf)
    clf = joblib.load(r'..\databases\results\deepBIQ\biq_model_spaq_1.obj')
    pred_y_test = clf.predict(X_test)

    # pred_y_test = clf.predict(X_test)
    # print('lcc:', cal_lcc(pred_y_test, y_test))

    evaluation(y_test, pred_y_test, draw_scatter=True)
    t = 0


if __name__ == '__main__':
    validation()




