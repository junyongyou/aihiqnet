import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from callbacks.callbacks import create_callbacks
from image_quality.sota.inception_koniq.ku import applications as apps

from image_quality.misc.imageset_handler import get_image_scores, get_image_score_from_groups
from image_quality.train.group_generator import GroupGenerator
from image_quality.train.plot_train import plot_history
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from callbacks.evaluation_callback import ModelEvaluationIQ
from image_quality.model_evaluation.evaluation import ModelEvaluation


def train_main(result_folder, model_name):
    metrics = 'mae'
    loss = 'mse'

    train_batch_size = 16
    test_batch_size = 16
    epochs = 120

    train_folders = [
        # r'...\databases\train\koniq_normal',]
                     r'...\databases\train\koniq_small',]
                     # r'...\databases\train\live']
    val_folders = [
        # r'...\databases\val\koniq_normal',]
                   r'...\databases\val\koniq_small',]
                   # r'...\databases\val\live']

    koniq_mos_file = r'...\databases\koniq10k_images_scores.csv'
    live_mos_file = r'...\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file)

    train_image_file_groups, train_score_groups = get_image_score_from_groups(train_folders, image_scores)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    train_generator = GroupGenerator(train_image_file_groups, train_score_groups, batch_size=train_batch_size,
                                     imagenet_pretrain=True)
    train_steps = train_generator.__len__()

    test_generator = GroupGenerator(test_image_file_groups, test_score_groups, batch_size=test_batch_size,
                                    image_aug=False,
                                    imagenet_pretrain=True)
    test_steps = test_generator.__len__()

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    evaluation_callback = ModelEvaluationIQ(test_image_files, test_scores, True,
                                            imagenet_pretrain=True)
    callbacks = create_callbacks(model_name, result_folder, other_callback=evaluation_callback, checkpoint=True,
                                 early_stop=False, metrics=metrics)
    total_train_steps = epochs * train_steps
    warmup_steps = 10 * train_steps
    lr_base = 1e-4/2
    warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr_base,
                                           total_steps=total_train_steps,
                                           warmup_learning_rate=0.0,
                                           warmup_steps=warmup_steps,
                                           hold_base_rate_steps=30 * train_steps,
                                           verbose=0)
    callbacks.append(warmup_lr)

    base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2, input_shape=(None, None, 3))
    head = apps.fc_layers(base_model.output, name='fc',
                          fc_sizes=[2048, 1024, 256, 1],
                          dropout_rates=[0.25, 0.25, 0.5, 0],
                          batch_norm=2)

    model = Model(inputs=base_model.input, outputs=head)

    optimizer = Adam(lr_base)

    # model.load_weights(r'..\databases\results\koncept_original\60_0.1081_0.2589_0.0555_0.1786.h5')
    # optimizer = Adam(5e-5)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
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

    model.save(os.path.join(result_folder, model_name + '.h5'))
    plot_history(model_history, result_folder, model_name)


def validation(weights_file):
    base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
    head = apps.fc_layers(base_model.output, name='fc',
                          fc_sizes=[2048, 1024, 256, 1],
                          dropout_rates=[0.25, 0.25, 0.5, 0],
                          batch_norm=2)

    model = Model(inputs=base_model.input, outputs=head)
    model.load_weights(weights_file)

    val_folders = [
                    r'...\databases\val\koniq_normal',]
                   # r'...\databases\val\koniq_small',]
                   # r'...\databases\train\live',
                   # r'...\databases\val\live']

    koniq_mos_file = r'...\databases\koniq10k_images_scores.csv'
    live_mos_file = r'...\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=True)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    evaluation = ModelEvaluation(model, test_image_files, test_scores, True,
                                 imagenet_pretrain=True)
    plcc, srcc, rmse = evaluation.__evaluation__()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    result_folder = r'...\databases\results\koncept_small'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    model_name = 'koncept'
    train_main(result_folder, model_name)

    # weights = r'...\databases\results\koncept_small\85_0.1352_0.2888_0.0454_0.1654.h5'
    # validation(weights)