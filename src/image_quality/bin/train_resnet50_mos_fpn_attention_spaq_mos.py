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


def train_main(args):
    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')

    result_folder = args['result_folder']
    model_name = 'triq_conv2D_all'

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

    if args['lr_base'] < 1e-4 / 2:
        model_name += '_finetune'
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
                             fpn_type=args['fpn_type'])

            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    else:
        model = phiq_net(n_quality_levels=args['n_quality_levels'],
                         naive_backbone=args['naive_backbone'],
                         backbone=args['backbone'],
                         fpn_type=args['fpn_type'])
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # model.summary()
    print('Load ImageNet weights')
    model.load_weights(args['weights'], by_name=True)

    # model.run_eagerly = True

    # imagenet_pretrain = True

    # Define train and validation data
    # train_images_scores, test_images_scores = load(open(r'..\databases\train_test_mos_spaq_koniq_live.pkl', 'rb'))
    train_images_scores, test_images_scores = load(open(r'..\databases\train_test_mos_spaq.pkl', 'rb'))
    train_generator = ResolutionGroupGenerator(train_images_scores, batch_size=args['batch_size'])
    train_steps = train_generator.__len__()

    validation_generator = ResolutionGroupGenerator(test_images_scores, batch_size=2, image_aug=False)
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


if __name__ == '__main__':
# def main():
    args = {}
    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['result_folder'] = r'..\databases\results\phiqnet_spaq_mos'
    args['n_quality_levels'] = 1

    args['backbone'] = 'resnet50'
    args['naive_backbone'] = False

    args['initial_epoch'] = 0

    args['lr_base'] = 1e-6#4/2
    args['lr_schedule'] = True
    args['batch_size'] = 4
    args['epochs'] = 120

    args['fpn_type'] = 'fpn'
    args['attention_module'] = True

    args['image_aug'] = True
    # args['weights'] = r'..\pretrained_weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    args['weights'] = r'..\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    args['do_finetune'] = True

    train_main(args)
