from image_quality.train.train import train_main


def main():
# if __name__ == '__main__':
    args = {}
    args['multi_gpu'] = 1
    args['gpu'] = 0

    # args['result_folder'] = r'..\databases\results\phiqnet_spaq_mos'
    args['result_folder'] = r'P:\600\60010\102300-Maintenance for Louse Counting Software\1. Arbeidsfiler\phiqnet_results\phiqnet_spaq_mos'
    args['n_quality_levels'] = 1

    # Choose between 'resnet50', 'densnet121', 'vgg16'
    args['backbone'] = 'resnet50'
    # args['backbone'] = 'densnet121'
    # args['backbone'] = 'vgg16'

    # Choose between False and True, default: False
    args['naive_backbone'] = False

    # Image and score must be provided
    args['images_scores_file'] = r'..\databases\train_val_test_spaq.pkl'
    args['image_folder'] = r'C:\fish_lice_dataset\image_quality_koniq10k\spaq\all'

    args['initial_epoch'] = 0

    args['lr_base'] = 1e-4/2
    args['lr_schedule'] = True
    args['batch_size'] = 16
    args['epochs'] = 100

    args['feature_fusion'] = True
    args['attention_module'] = True

    args['image_aug'] = True

    # Depending on which backbone is used, choose the corresponding ImageNet pretrained weights file, set to None is no pretrained weights to be used.
    # args['weights'] = r'..\pretrained_weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'..\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    args['weights'] = r'C:\Users\junyong\Downloads\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'C:\Users\junyong\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = r'C:\Users\junyong\Downloads\densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # args['weights'] = None

    args['do_finetune'] = True

    train_main(args)
