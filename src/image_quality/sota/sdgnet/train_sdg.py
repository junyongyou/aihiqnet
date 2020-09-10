import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from callbacks.callbacks import create_callbacks

from image_quality.misc.imageset_handler import get_image_scores, get_image_score_from_groups
from image_quality.sota.sdgnet.group_generator_sgd import GroupGenerator
from image_quality.train.plot_train import plot_history
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from image_quality.sota.sdgnet.evaluation_callback_sgd import ModelEvaluationIQ
from image_quality.sota.sdgnet.evaluation_sgd import ModelEvaluation
from image_quality.sota.sdgnet.modelfinal import TVdist, SGDNet

import os


def train_main(result_folder, model_name):
    train_batch_size = 4
    test_batch_size = 4
    epochs = 120

    train_folders = [r'...\databases\train\koniq_normal',]
                     # r'...\databases\train\koniq_small',]
                     # r'...\databases\train\live']
    val_folders = [r'...\databases\val\koniq_normal',]
                   # r'...\databases\val\koniq_small',]
                   # r'...\databases\val\live']s

    koniq_mos_file = r'...\databases\koniq10k_images_scores.csv'
    live_mos_file = r'...\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file)

    train_image_file_groups, train_score_groups = get_image_score_from_groups(train_folders, image_scores)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    saliency_folder = r'...\databases\predictions_sam_koniq10k'
    train_generator = GroupGenerator(train_image_file_groups, train_score_groups, saliency_folder, batch_size=train_batch_size,
                                     imagenet_pretrain=False)
    train_steps = train_generator.__len__()

    test_generator = GroupGenerator(test_image_file_groups, test_score_groups, saliency_folder, batch_size=test_batch_size,
                                    image_aug=False,
                                    imagenet_pretrain=False)
    test_steps = test_generator.__len__()

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    evaluation_callback = ModelEvaluationIQ(test_image_files, test_scores, True, saliency_folder, imagenet_pretrain=False)
    callbacks = create_callbacks(model_name, result_folder, evaluation_callback, checkpoint=True, early_stop=False,
                                 metrics=None)

    warmup_epochs = 10
    lr_base = 1e-4 / 2
    total_train_steps = epochs * train_steps
    warmup_steps = warmup_epochs * train_steps
    warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr_base,
                                           total_steps=total_train_steps,
                                           warmup_learning_rate=0.0,
                                           warmup_steps=warmup_steps,
                                           hold_base_rate_steps=30*train_steps,
                                           verbose=0)
    callbacks.append(warmup_lr)
    model_name += '_lr'

    saliency = 'input'
    weights = r'...\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = SGDNet(basemodel='resnet', saliency=saliency, CA=True, fixed=False, img_cols=None, img_rows=None, weights=weights)

    optimizer = Adam(lr_base)
    alpha = 0.25
    if saliency == 'output':
        model.compile(optimizer=optimizer, loss=['mae', TVdist], loss_weights=[1.0 / (1 + alpha), alpha / (1 + alpha)])
    else:
        model.compile(optimizer=optimizer, loss=['mae'])

    model.summary()
    model_history = model.fit(x=train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_steps,
                              validation_data=test_generator,
                              validation_steps=test_steps,
                              verbose=1,
                              shuffle=False,
                              callbacks=callbacks,
                              initial_epoch=0)

    # model.save(os.path.join(result_folder, model_name + '.h5'))
    plot_history(model_history, result_folder, model_name)


def validation(weights_file):
    model = SGDNet(basemodel='resnet', saliency='input', CA=True, fixed=False, img_cols=None, img_rows=None)
    model.load_weights(weights_file)

    val_folders = [#r'...\databases\val\koniq_normal',]
                   r'...\databases\val\koniq_small',]
                   # r'...\databases\val\live']

    koniq_mos_file = r'...\databases\koniq10k_images_scores.csv'
    live_mos_file = r'...\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=True)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    saliency_folder = r'...\databases\predictions_sam_koniq10k'
    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    evaluation = ModelEvaluation(model, test_image_files, test_scores, True, saliency_folder,
                                 imagenet_pretrain=False)
    plcc, srcc, rmse = evaluation.__evaluation__()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    result_folder = r'...\databases\results\sdg_small'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    model_name = 'mos_sdg'
    train_main(result_folder, model_name)

    # weights_file = r'..\databases\results\mos_sdg\67_0.1745_0.2471.h5'
    # weights_file = r'..\databases\results\mos_sdg_normal\50_0.2247_0.2288.h5'
    # validation((weights_file))
