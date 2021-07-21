import os
import numpy as np
import glob
from pickle import load
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from image_quality.models.image_quality_model import phiq_net
from callbacks.callbacks import create_callbacks
from image_quality.train.plot_train import plot_history
from image_quality.train.resolution_group_generator import ResolutionGroupGenerator
from callbacks.evaluation_callback_generator import ModelEvaluationIQGenerator
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from image_quality.model_evaluation.model_evaluation import ModelEvaluation


def identify_best_weights(result_folder, history, best_plcc):
    pos = np.where(history['plcc'] == best_plcc)[0][0]

    pos_loss = '{}_{:.4f}'.format(pos + 1, history['loss'][pos])
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        weight_file = os.path.basename(all_weights_file)
        if weight_file.startswith(pos_loss):
            best_weights_file = all_weights_file
            return best_weights_file
    return None


def remove_non_best_weights(result_folder, best_weights_files):
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        if all_weights_file not in best_weights_files:
            os.remove(all_weights_file)


def get_image_list_path(image_folder, image_score_list, n_quality_levels, do_normalization=False):
    images_scores = []
    for image_score in image_score_list:
        content = image_score.split(',')
        if len(content) > 2:
            if n_quality_levels > 1:
                scores_softmax = np.array([float(score) for score in content[1: 6]])
                score = [score_softmax / scores_softmax.sum() for score_softmax in scores_softmax]

            else:
                score = float(content[-1])
        else:
            if do_normalization:
                score = float(content[1]) / 25 + 1
            else:
                score = float(content[1])
        images_scores.append('{};{}'.format(os.path.join(image_folder, content[0]), score))
    return images_scores


def check_args(args):
    if 'result_folder' not in args:
        exit('Result folder must be specified')
    if 'images_scores_file' not in args:
        exit('Image_score file must be specified')
    if 'image_folder' not in args:
        args['image_folder'] = None
        print('WARN: check the image paths are specified in images_scores_file, otherwise image folder must be specified')

    if 'n_quality_levels' not in args:
        exit('Number of quality levels (1 or 5) must be specified')

    if 'epochs' not in args:
        args['epochs'] = 100
    if 'lr_base' not in args:
        args['lr_base'] = 1e-4 / 2
    if 'naive_backbone' not in args:
        args['naive_backbone'] = False
    if 'backbone' not in args:
        args['backbone'] = 'resnet50'
    if 'weights' not in args:
        args['weights'] = None
    if 'initial_epoch' not in args:
        args['initial_epoch'] = 0
    if 'feature_fusion' not in args:
        args['feature_fusion'] = True
    if 'attention_module' not in args:
        args['attention_module'] = True
    if 'freeze_backbone' not in args:
        args['freeze_backbone'] = False
    if 'lr_schedule' not in args:
        args['lr_schedule'] = True
    if 'batch_size' not in args:
        args['batch_size'] = 16
    if 'image_aug' not in args:
        args['image_aug'] = True
    if 'do_finetune' not in args:
        args['do_finetune'] = True
    if 'multi_gpu' not in args:
        args['multi_gpu'] = 0
    if 'gpu' not in args:
        args['gpu'] = 0

    return args


def train_main(args):
    args = check_args(args)

    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')

    result_folder = args['result_folder']
    model_name = args['backbone']

    # Define loss function according to prediction objective (score distribution or MOS)
    if args['n_quality_levels'] > 1:
        using_single_mos = False
        loss = 'categorical_crossentropy'
        metrics = None
        model_name += '_distribution'
    else:
        using_single_mos = True
        metrics = None
        loss = 'mse'
        model_name += '_mos'

    if args['naive_backbone']:
        model_name += '_naive'
    if args['weights'] is None:
        model_name += '_nopretrain'
    if not args['image_aug']:
        model_name += '_no_imageaug'

    # Create PHIQnet model
    optimizer = Adam(args['lr_base'])

    if args['multi_gpu'] > 0:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = phiq_net(n_quality_levels=args['n_quality_levels'],
                             naive_backbone=args['naive_backbone'],
                             backbone=args['backbone'],
                             feature_fusion=args['feature_fusion'],
                             attention_module=args['attention_module'])

            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    else:
        model = phiq_net(n_quality_levels=args['n_quality_levels'],
                         naive_backbone=args['naive_backbone'],
                         backbone=args['backbone'],
                         feature_fusion=args['feature_fusion'],
                         attention_module=args['attention_module'])
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # model.summary()
    # Load Imagenet pretrained weights or existing weights for fine-tune
    if args['weights'] is not None:
        print('Load weights: {}'.format(args['weights']))
        model.load_weights(args['weights'], by_name=True)

    # model.run_eagerly = True

    if args['weights'] is not None and ('resnet' in args['backbone'] or args['backbone'] == 'inception'):
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    # Define train and validation data
    with open(args['images_scores_file'], 'rb') as f:
        train_images_scores, val_images_scores, test_images_scores = load(f)
    if args['image_folder']:
        train_images_scores = get_image_list_path(args['image_folder'], train_images_scores, args['n_quality_levels'], do_normalization=True)
        val_images_scores = get_image_list_path(args['image_folder'], val_images_scores, args['n_quality_levels'], do_normalization=True)
        test_images_scores = get_image_list_path(args['image_folder'], test_images_scores, args['n_quality_levels'], do_normalization=True)

    train_generator = ResolutionGroupGenerator(train_images_scores, batch_size=args['batch_size'], imagenet_pretrain=imagenet_pretrain)
    train_steps = train_generator.__len__()

    validation_generator = ResolutionGroupGenerator(val_images_scores, batch_size=2, imagenet_pretrain=imagenet_pretrain, image_aug=False)
    validation_steps = validation_generator.__len__()

    evaluation_callback = ModelEvaluationIQGenerator(validation_generator,
                                                     using_single_mos)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Create callbacks including evaluation and learning rate scheduler
    callbacks = create_callbacks(model_name,
                                 result_folder,
                                 evaluation_callback,
                                 checkpoint=True,
                                 early_stop=True,
                                 metrics=metrics)

    warmup_epochs = 10
    if args['lr_schedule']:
        total_train_steps = args['epochs'] * train_steps
        warmup_steps = warmup_epochs * train_steps
        warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=args['lr_base'],
                                               total_steps=total_train_steps,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=30 * train_steps,
                                               verbose=1)
        callbacks.append(warmup_lr)

    # Define optimizer and train
    model_history = model.fit(x=train_generator,
                              epochs=args['epochs'],
                              steps_per_epoch=train_steps,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              verbose=1,
                              shuffle=False,
                              callbacks=callbacks,
                              initial_epoch=args['initial_epoch'],
                              )
    # model.save(os.path.join(result_folder, model_name + '.h5'))
    # plot_history(model_history, result_folder, model_name)

    best_weights_file = identify_best_weights(result_folder, model_history.history, callbacks[3].best)
    remove_non_best_weights(result_folder, [best_weights_file])

    evaluation_testset = ModelEvaluation(model, using_single_mos=using_single_mos, imagenet_pretrain=imagenet_pretrain)
    evaluation_testset.__evaluation__(best_weights_file, test_images_scores, evaluation_name='testset_basetrain', result_folder=result_folder, draw_scatter=True)

    # do fine-tuning
    if args['do_finetune'] and best_weights_file:
        print('Finetune...')
        del (callbacks[-1])
        model.load_weights(best_weights_file)
        finetune_lr = 1e-6
        if args['lr_schedule']:
            warmup_lr_finetune = WarmUpCosineDecayScheduler(learning_rate_base=finetune_lr,
                                                            total_steps=total_train_steps,
                                                            warmup_learning_rate=0.0,
                                                            warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=10 * train_steps,
                                                            verbose=1)
            callbacks.append(warmup_lr_finetune)
        finetune_optimizer = Adam(finetune_lr)
        model.compile(loss=loss, optimizer=finetune_optimizer, metrics=[metrics])

        finetune_model_history = model.fit(x=train_generator,
                                  epochs=args['epochs'],
                                  steps_per_epoch=train_steps,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  verbose=1,
                                  shuffle=False,
                                  callbacks=callbacks,
                                  initial_epoch=args['initial_epoch'],
                                  )

        best_weights_file_finetune = identify_best_weights(result_folder, finetune_model_history.history, callbacks[3].best)
        remove_non_best_weights(result_folder, [best_weights_file, best_weights_file_finetune])
        evaluation_testset.__evaluation__(best_weights_file_finetune, test_images_scores, evaluation_name='testset_finetune',
                                          result_folder=result_folder, draw_scatter=True)
