"""
Main function to build PHIQnet.
"""
from image_quality.layers.fusion import fusion_layer, no_fusion
from backbone.ResNest import ResNest
from tensorflow.keras.layers import Input, Dense, Average, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from image_quality.models.prediction_model_contrast_sensitivity import channel_spatial_attention
from backbone.resnet50 import ResNet50
from backbone.resnet_family import ResNet18
from backbone.resnet_feature_maps import ResNet152v2, ResNet152
from backbone.vgg16 import VGG16
from backbone.densenet import DenseNet121
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf


def phiq_net(n_quality_levels, input_shape=(None, None, 3), naive_backbone=False, backbone='resnet50', feature_fusion=True,
             attention_module=True):
    """
    Build PHIQnet
    :param n_quality_levels: 1 for MOS prediction and 5 for score distribution
    :param input_shape: image input shape, keep as unspecifized
    :param naive_backbone: flag to use backbone only, i.e., without neck and head, if set to True
    :param backbone: backbone networks (resnet50/18/152v2, resnest, vgg16, etc.)
    :param feature_fusion: flag to use or not feature fusion
    :param attention_module: flag to use or not attention module
    :return: PHIQnet model
    """
    inputs = Input(shape=input_shape)
    n_classes = None
    return_feature_maps = True
    if naive_backbone:
        n_classes = 1
        return_feature_maps = False
    fc_activation = None
    verbose = False
    if backbone == 'resnest50':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=32, avg_down=True, avd=True, avd_first=False,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnest34':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=16, avg_down=True, avd=True, avd_first=False, using_basic_block=True,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnest18':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[2, 2, 2, 2], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=16, avg_down=True, avd=True, avd_first=False, using_basic_block=True,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnet50':
        backbone_model = ResNet50(inputs,
                                  return_feature_maps=return_feature_maps)
    elif backbone == 'resnet18':
        backbone_model = ResNet18(input_tensor=inputs,
                                  weights=None,
                                  include_top=False)
    elif backbone == 'resnet152v2':
        backbone_model = ResNet152v2(inputs)
    elif backbone == 'resnet152':
        backbone_model = ResNet152(inputs)
    elif backbone == 'vgg16':
        backbone_model = VGG16(inputs)
    elif backbone == 'densnet121':
        backbone_model = DenseNet121(inputs, return_feature_maps=return_feature_maps)
    else:
        raise NotImplementedError

    if naive_backbone:
        backbone_model.summary()
        return backbone_model

    C2, C3, C4, C5 = backbone_model.outputs
    pyramid_feature_size = 256
    if feature_fusion:
        fpn_features = fusion_layer(C2, C3, C4, C5, feature_size=pyramid_feature_size)
    else:
        fpn_features = no_fusion(C2, C3, C4, C5, feature_size=pyramid_feature_size)

    PF = []
    for i, P in enumerate(fpn_features):
        if attention_module:
            PF.append(channel_spatial_attention(P, n_quality_levels, 'P{}'.format(i)))
        else:
            outputs = GlobalAveragePooling2D(name='avg_pool_{}'.format(i))(P)
            if n_quality_levels > 1:
                outputs = Dense(n_quality_levels, activation='softmax', name='fc_prediction_{}'.format(i))(outputs)
            else:
                outputs = Dense(n_quality_levels, activation='linear', name='fc_prediction_{}'.format(i))(outputs)
            PF.append(outputs)
    outputs = Average(name='PF_average')(PF)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # input_shape = [None, None, 3]
    input_shape = [768, 1024, 3]
    # input_shape = [500, 500, 3]
    # model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet152v2')
    model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet50')
    # model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='vgg16')
    # model = adiq_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet18', attention_module=True)
