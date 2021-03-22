from image_quality.models.image_quality_model import phiq_net
from image_quality.model_evaluation.evaluation_spaq import ModelEvaluation
import os
import tensorflow as tf


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


def val_main(args):
    if args['n_quality_levels'] > 1:
        using_single_mos = False
    else:
        using_single_mos = True

    if args['weights'] is not None and ('resnet' in args['backbone'] or args['backbone'] == 'inception'):
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    val_folders = [r'\CID2013']
    # val_folders = [r'F:\SPAG_image_quality_dataset\SPAQ\TestImage']
    cid_mos_file = r'CID2013\image_mos.txt'
    # spaq_mos_file = r'..databases	rainspaq\image_mos.csv'
    image_files, image_scores = get_image_scores(val_folders, cid_mos_file)

    model = phiq_net(n_quality_levels=args['n_quality_levels'],
                     naive_backbone=args['naive_backbone'],
                     backbone=args['backbone'],
                     fpn_type=args['fpn_type'])
    model.load_weights(args['weights'])

    evaluation = ModelEvaluation(model, image_files, image_scores, using_single_mos,
                                 imagenet_pretrain=imagenet_pretrain)
    result_file = None#r'..\databases\spag\result.csv'
    plcc, srcc, rmse = evaluation.__evaluation__(result_file)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    args = {}
    # args['result_folder'] = r'..\databases\results'
    args['n_quality_levels'] = 1
    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['fpn_type'] = 'fpn'
    args['weights'] = r'..\databases\experiments\koniq_all\resnet50_mos_attention_fpn_finetune\91_0.0003_0.0851.h5'

    val_main(args)