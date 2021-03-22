import argparse
import os
import shutil
import time
from image_quality.misc.imageset_handler import get_image_scores, get_image_score_from_groups

import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
import scipy.io as sio
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from pickle import dump, load
from common_model import *


# svr_save_path = './svr_mode.pkl'
# svr_process_path = './svr_process.pkl'
feature_mode_path = '../trained_models/model_best.pth.tar'
image_dir = './ChallengeDB_release/Images'
matfn = './ChallengeDB_release/Data/AllMOS_release.mat'
mat_img_name = './ChallengeDB_release/Data/AllImages_release.mat'


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

def generate_data():
    train_folders = [
        r'..\databases\train\koniq_normal',
        r'..\databases\train\koniq_small',
        r'..\databases\train\live']
    val_folders = [
        r'..\databases\val\koniq_normal',
        r'..\databases\val\koniq_small',
        r'..\databases\val\live']

    koniq_mos_file = r'..\databases\koniq10k_images_scores.csv'
    live_mos_file = r'..\databases\live_wild\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file)

    train_image_file_groups, train_score_groups = get_image_score_from_groups(train_folders, image_scores)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    train_image_files = []
    train_scores = []
    for train_image_file_group, train_score_group in zip(train_image_file_groups, train_score_groups):
        train_image_files.extend(train_image_file_group)
        train_scores.extend(train_score_group)

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    cal_features(train_image_files)
    cal_features(test_image_files)


def img_dataset(feature_model):
    mos_data = sio.loadmat(matfn)
    mos = mos_data['AllMOS_release']
    img_name_data = sio.loadmat(mat_img_name)
    img_name = img_name_data['AllImages_release']

    normalize = get_imagenet_normalize()
    img_transform = transforms.Compose([transforms.ToTensor(), normalize])
    #img_num = 1169
    img_num = mos.shape[1]
    idx_arry = np.arange(0, img_num)
    np.random.shuffle(idx_arry)

    X = np.array([])
    Y = np.array([])

    crop_w = 224
    crop_h = 224
    img_w = 0
    img_h = 0
    crop_num_w = 5
    crop_num_h = 5

    for i, idx in enumerate(idx_arry):
        img_file_path = os.path.join(image_dir, img_name[idx][0][0])
        img_mos_score = mos[0, idx]
        print(i, ' process: ', img_file_path)
        crop_imgs = np.array([])
        crop_out = None
        img = Image.open(img_file_path)
        img_w, img_h = img.size
        crop_box = get_crop_box(img_w, img_h, crop_w, crop_h, crop_num_w, crop_num_h)
        for box in crop_box:
            crop_imgs = np.append(crop_imgs, img_transform(img.crop(box)).numpy())
        crop_imgs = crop_imgs.reshape(crop_num_w * crop_num_h, 3, 224, 224)
        crop_imgs = torch.from_numpy(crop_imgs).float()
        crop_out = feature_model.extract_feature(crop_imgs)
        crop_out = np.average(crop_out, axis=0)

        X = np.append(X, crop_out)
        Y = np.append(Y, img_mos_score)
    X = X.reshape(-1, 4096)

    print(X.shape)
    print(Y.shape)

    return X, Y


def collect_features_scores(image_files, scores):
    X = np.array([])
    Y = np.array([])

    for image_file, score in zip(image_files, scores):
        extname = os.path.splitext(os.path.basename(image_file))[1]
        feature_path = image_file.replace(r'..\databases',
                                                r'..\databases\AlexNet_features').replace(
            extname, '.npy')
        feature = np.load(feature_path)
        X = np.append(X, feature)
        Y = np.append(Y, score)
    X = X.reshape(-1, 4096)
    return X, Y


def main():
    # feature_model = FeatureMode(feature_mode_path)
    # data_x, data_y = img_dataset(feature_model)
    train_folders = [
        r'..\databases\train\koniq_normal',
        r'..\databases\train\koniq_small',
        r'..\databases\train\live']
    val_folders = [
        r'..\databases\val\koniq_normal',
        r'..\databases\val\koniq_small',
        r'..\databases\val\live']

    koniq_mos_file = r'..\databases\koniq10k_images_scores.csv'
    live_mos_file = r'..\databases\live_wild\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file)

    train_image_file_groups, train_score_groups = get_image_score_from_groups(train_folders, image_scores)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    train_image_files = []
    train_scores = []
    for train_image_file_group, train_score_group in zip(train_image_file_groups, train_score_groups):
        train_image_files.extend(train_image_file_group)
        train_scores.extend(train_score_group)

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    X_train, y_train = collect_features_scores(train_image_files, train_scores)
    X_test, y_test = collect_features_scores(test_image_files, test_scores)

    scaler_x = preprocessing.StandardScaler().fit(X_train)
    with (open(r'..\databases\results\deepBIQ\scaler_all.obj', 'wb')) as scaler_file:
        dump(scaler_x, scaler_file)
    # scaler_x = load(open(r'..\databases\results\deepBIQ\scaler.obj', 'rb'))
    joblib.dump(scaler_x, r'..\databases\results\deepBIQ\scaler_all_1.obj')
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)

    # X_train, X_test, y_train, y_test = train_test_split(train_x, data_y, test_size=0.2, random_state=0)
    print('------------')
    print('training svr model ......')

    # parameters = {"C": [1e1, 1e2, 1e3], "gamma": [0.00030, 0.00020, 0.00010],
    #               "epsilon": [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]}
    #               "epsilon": [10.0, 1.0, 0.1, 0.01]}
    parameters = {"C": [1e1, 1e2, 1e3], "gamma": [0.00025, 0.00020, 0.00015, 0.00010],
                  "epsilon": [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]}
    # clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1, epsilon=0.01), cv=3, param_grid=parameters, n_jobs=20, verbose=5)
    clf = RandomizedSearchCV(SVR(kernel='rbf', gamma=0.1, epsilon=0.01), cv=3, param_distributions=parameters, n_jobs=20, verbose=5)
    clf.fit(X_train, y_train)

    #best score
    print("Best score: %0.3f" % clf.best_score_)
    print("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    clf = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], epsilon=best_parameters['epsilon'])
    clf.fit(X_train, y_train)

    with open(r'..\databases\results\deepBIQ\biq_model_all.obj', 'wb') as clf_file:
        dump(clf, clf_file)
    joblib.dump(clf, r'..\databases\results\deepBIQ\biq_model_all_1.obj')

    pred_y_test = clf.predict(X_test)
    print('lcc:', cal_lcc(pred_y_test, y_test))


if __name__ == '__main__':
    # f = open(r'..\databases\results\deepBIQ\scaler.obj', 'rb')
    # scaler_x = load(open(r'..\databases\results\deepBIQ\scaler.obj', 'rb'))
    main()

    # mos_data = sio.loadmat(r'D:\Downloads\ChallengeDB_release\ChallengeDB_release\Data\AllMOS_release.mat')
    # mos = mos_data['AllMOS_release']
    # t = 0
    # generate_data()




