from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import zip, map, range
from past.utils import old_div
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import reduce
from .model_helper import *
import tensorflow.keras.backend as K

import tensorflow.keras as keras
import sys
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.nasnet import NASNetMobile

source_module = {
                 InceptionV3:       keras.applications.inception_v3,
                 DenseNet201:       keras.applications.densenet,
                 ResNet50:          keras.applications.resnet50,
                 InceptionResNetV2: keras.applications.inception_resnet_v2,
                 VGG16:             keras.applications.vgg16,
                 NASNetMobile:      keras.applications.nasnet
                }

# correspondences between CNN model name and pre-processing function
process_input = {
                 InceptionV3:       keras.applications.inception_v3.preprocess_input,
                 DenseNet201:       keras.applications.densenet.preprocess_input,
                 ResNet50:          keras.applications.resnet50.preprocess_input,
                 InceptionResNetV2: keras.applications.inception_resnet_v2.preprocess_input,
                 VGG16:             keras.applications.vgg16.preprocess_input,
                 NASNetMobile:      keras.applications.nasnet.preprocess_input
                }

def fc_layers(input_layer,
              name               = 'pred',
              fc_sizes           = [2048, 1024, 256, 1],
              dropout_rates      = [0.25, 0.25, 0.5, 0],
              batch_norm         = False,
              l2_norm_inputs     = False,              
              activation         = 'relu',
              initialization     = 'he_normal',
              out_activation     = 'linear',
              test_time_dropout  = False,
              **fc_params):
    """
    Add a standard fully-connected (fc) chain of layers (functional Keras interface)
    with dropouts on top of an input layer. Optionally batch normalize, add regularizers
    and an output activation.

    e.g. default would look like dense(2048) > dropout

    * input_layer: input layer to the chain
    * name: prefix to each layer in the chain
    * fc_sizes: list of number of neurons in each fc-layer
    * dropout_rates: list of dropout rates for each fc-layer
    * batch_norm: 0 (False) = no batch normalization (BN),
                  1 = do BN for all, 2 = do for all except the last
    * l2_norm_inputs: normalize the `input_layer` with L2_norm
    * kernel_regularizer: optional regularizer for each fc-layer
    * out_activation: activation added to the last fc-layer
    :return: output layer of the chain
    """
    x = input_layer
    if l2_norm_inputs:
        x = Lambda(lambda x: tf.nn.l2_normalize(x, 1))(input_layer)

    assert dropout_rates is None or (len(fc_sizes) == len(dropout_rates)),\
           'Each FC layer should have a corresponding dropout rate'

    if activation.lower() == 'selu':
        dropout_call = AlphaDropout
    else:
        dropout_call = Dropout
        
    for i in range(len(fc_sizes)):
        if i < len(fc_sizes)-1:
            act = activation
            layer_type = 'fc%d' % i
        else:
            act  = out_activation
            layer_type = 'out'
        x = Dense(fc_sizes[i], activation=act, 
                  name='%s_%s' % (name, layer_type),
                  kernel_initializer=initialization, 
                  **fc_params)(x)
        if batch_norm > 0 and i < ( len(fc_sizes)-(batch_norm-1) ):
            x = BatchNormalization(name='%s_bn%d' % (name, i))(x)
        if dropout_rates is not None and dropout_rates[i] > 0:
            do_call = dropout_call(dropout_rates[i], name = '%s_do%d' % (name, i))            
            if test_time_dropout:
                x = do_call(x, training=True)
            else:
                x = do_call(x)
                
    return x

def conv2d_bn(x, filters, num_row, num_col, padding='same',
              strides=(1, 1), name=None):
    """
    Utility function to apply conv + BN.
    
    * x:       input tensor.
    * filters: filters in `Conv2D`.
    * num_row: height of the convolution kernel.
    * num_col: width of the convolution kernel.
    * padding: padding mode in `Conv2D`.
    * strides: strides in `Conv2D`.
    * name:    name of the ops; will become `name + '_conv'`
               for the convolution and `name + '_bn'` for the
               batch norm layer.
    :return:   output tensor after applying `Conv2D` and `BatchNormalization`.

    Source: InceptionV3 Keras code
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters, (num_row, num_col),
               strides=strides, padding=padding,
               use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def inception_block(x, size=768, name=''):
    channel_axis = 3
    
    branch1x1 = conv2d_bn(x, size, 1, 1, name=name+'branch_1x1')

    branch3x3 = conv2d_bn(x, size, 1, 1, name=name+'3x3_1x1')
    branch3x3_1 = conv2d_bn(branch3x3, old_div(size,2), 1, 3, name=name+'3x3_1x3')
    branch3x3_2 = conv2d_bn(branch3x3, old_div(size,2), 3, 1, name=name+'3x3_3x1')
    branch3x3 = concatenate(
        [branch3x3_1, branch3x3_2],
        axis=channel_axis,
        name=name+'branch_3x3')

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), 
                                   padding='same', 
                                   name=name+'avg_pool_2d')(x)
    branch_pool = conv2d_bn(branch_pool, size, 1, 1, 
                            name=name+'branch_pool')
    
    y = concatenate(
        [branch1x1, branch3x3, branch_pool],
        axis=channel_axis,
        name=name+'mixed_final')
    return y

def model_inception_multigap(input_shape=(224, 224, 3), return_sizes=False,
                             indexes=list(range(11)), name = ''):
    """
    Build InceptionV3 multi-GAP model, that extracts narrow MLSP features.
    Relies on `get_inception_gaps`.

    * input_shape: shape of the input images
    * return_sizes: return the sizes of each layer: (model, gap_sizes)
    * indexes: indices to use from the usual GAPs
    * name: name of the model
    :return: model or (model, gap_sizes)
    """
    print('Loading InceptionV3 multi-gap with input_shape:', input_shape)

    model_base = InceptionV3(weights     = 'imagenet', 
                             include_top = False, 
                             input_shape = input_shape)
    print('Creating multi-GAP model')
    
    gap_name = name + '_' if name else ''

    feature_layers = [model_base.get_layer('mixed%d' % i) 
                      for i in indexes]
    gaps = [GlobalAveragePooling2D(name=gap_name+"gap%d" % i)(l.output)
            for i, l in zip(indexes, feature_layers)]
    concat_gaps = Concatenate(name=gap_name+'concat_gaps')(gaps)

    model = Model(inputs  = model_base.input,
                  outputs = concat_gaps)
    if name:
        model.name = name
    
    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
        return (model, gap_sizes)
    else:
        return model


def model_inceptionresnet_multigap(input_shape=(224, 224, 3),
                                   return_sizes=False):
    """
    Build InceptionResNetV2 multi-GAP model, that extracts narrow MLSP features.

    * input_shape: shape of the input images
    * return_sizes: return the sizes of each layer: (model, gap_sizes)
    :return: model or (model, gap_sizes)
    """
    print('Loading InceptionResNetV2 multi-gap with input_shape:', input_shape)

    model_base = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=input_shape)
    print('Creating multi-GAP model')
    
    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    gaps = [GlobalAveragePooling2D(name="gap%d" % i)(l.output)
            for i, l in enumerate(feature_layers)]
    concat_gaps = Concatenate(name='concatenated_gaps')(gaps)

    model = Model(inputs=model_base.input, outputs=concat_gaps)

    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
        return (model, gap_sizes)
    else:
        return model
    
def model_inception_pooled(input_shape=(None, None, 3), indexes=list(range(11)),
                           pool_size=(5, 5), name='', return_sizes=False):
    """
    Returns the wide MLSP features, spatially pooled, from InceptionV3.
    Similar to `model_inception_multigap`.

    * input_shape: shape of the input images
    * indexes: indices to use from the usual pools
    * pool_size: spatial extend of the MLSP features
    * name: name of the model
    * return_sizes: return the sizes of each layer: (model, pool_sizes)
    :return: model or (model, pool_sizes)
    """
    print('Loading InceptionV3 multi-pooled with input_shape:', input_shape)
    model_base = InceptionV3(weights     = 'imagenet', 
                             include_top = False, 
                             input_shape = input_shape)
    print('Creating multi-pooled model')
    
    ImageResizer = Lambda(lambda x: tf.image.resize_area(x, pool_size),
                          name='feature_resizer')

    feature_layers = [model_base.get_layer('mixed%d' % i) for i in indexes]
    pools = [ImageResizer(l.output) for l in feature_layers]
    conc_pools = Concatenate(name='conc_pools', axis=3)(pools)

    model = Model(inputs  = model_base.input, 
                  outputs = conc_pools)
    if name: model.name = name

    if return_sizes:
        pool_sizes = [[np.int32(x) for x in f.get_shape()[1:]] for f in pools]
        return model, pool_sizes
    else:
        return model
    
def model_inceptionresnet_pooled(input_shape=(None, None, 3), indexes=list(range(43)),
                                 pool_size=(5, 5), name='', return_sizes=False):
    """
    Returns the wide MLSP features, spatially pooled, from InceptionResNetV2.

    * input_shape: shape of the input images
    * indexes: indices of the modules to use
    * pool_size: spatial extend of the MLSP features
    * name: name of the model
    * return_sizes: return the sizes of each layer: (model, pool_sizes)
    :return: model or (model, pool_sizes)
    """
    
    print('Loading InceptionResNetV2 multi-pooled with input_shape:', input_shape)
    model_base = InceptionResNetV2(weights     = 'imagenet', 
                                   include_top = False, 
                                   input_shape = input_shape)
    print('Creating multi-pooled model')
    
    ImageResizer = Lambda(lambda x: tf.image.resize_area(x, pool_size),
                          name='feature_resizer') 

    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    feature_layers = [feature_layers[i] for i in indexes]
    pools = [ImageResizer(l.output) for l in feature_layers]
    conc_pools = Concatenate(name='conc_pools', axis=3)(pools)

    model = Model(inputs  = model_base.input, outputs = conc_pools)
    if name: model.name = name

    if return_sizes:
        pool_sizes = [[np.int32(x) for x in f.get_shape()[1:]] for f in pools]
        return model, pool_sizes
    else:
        return model    


# ------------------
# RATING model utils
# ------------------

def test_rating_model(helper, ids_test=None, 
                      output_layer=None, output_column=None, 
                      groups=1, remodel=False, show_plot=True):
    """
    Test rating model performance. The output of the mode is assumed to be
    either a single score, or distribution of scores (can be a histogram).

    * helper:   ModelHelper object that contains the trained model
    * ids_test: optionally provide another set of data instances, replacing 
                those in `helper.ids`
    * output_layer: the rating layer, if more than out output exists
                    if output_layer is None, we assume there is a single rating output
    * output_column: the column in ids that corresponds to the output
    * groups: if a number: repetitions of the testing procedure,
                           after which the results are averaged
              if list of strings: group names to repeat over,
                                  they are assumed to be different augmentations
    * remodel: change structure of the model when changing `output_layer`, or not
    * show_plot: plot results vs ground-truth
    :return: (y_true, y_pred, SRCC, PLCC)
    """
    print('Model outputs:', helper.model.output_names)
    if output_column is not None: 
        print('Output column:', output_column)
    if ids_test is None: ids_test = helper.ids[helper.ids.set=='test']

    if isinstance(groups, numbers.Number):
        test_gen = helper.make_generator(ids_test, 
                                         shuffle = False,
                                         fixed_batches = False)
        y_pred = helper.predict(test_gen, repeats=groups, remodel=remodel,
                                output_layer=output_layer)
        groups_list = list(range(groups))
    else:
        if isinstance(groups[0], (list, tuple, str)):
            groups_list = groups
        else:
            groups_list = list(map(str, groups))
        y_pred = []
        print('Predicting on groups:')
        for group in groups_list:
            print(group, end=' ')
            test_gen = helper.make_generator(ids_test, shuffle=False,
                                             fixed_batches = False,
                                             random_group  = False,
                                             group_names   = force_list(group))
            y_pred.append(helper.predict(test_gen, repeats=1, remodel=remodel,
                                         output_layer=output_layer))
        print()
    
    if isinstance(y_pred, list):
        y_pred = old_div(reduce(lambda x, y: (x+y), y_pred), len(y_pred))

    y_pred = np.squeeze(y_pred)

    if y_pred.ndim == 2: 
        # for distributions
        print('Testing distributions')
        outputs = helper.gen_params.outputs
        y_pred = dist2mos(y_pred, scale=np.arange(1, len(outputs)+1))
        y_test = np.array(ids_test.loc[:, outputs])
        y_test = dist2mos(y_test, scale=np.arange(1, len(outputs)+1))
    else:                 
        # for single prediction
        print('Testing single prediction')
        if output_column is None:
            output = force_tuple(helper.gen_params.outputs)[0]
        else:
            output = output_column
        y_test = np.array(ids_test.loc[:,output])
        y_test = y_test.flatten()

    # in case the last batch was not used, and dataset size
    # is not a multiple of batch_size
    y_test = y_test[:len(y_pred)]
    
    SRCC_test = round(srocc(y_pred, y_test), 3)
    PLCC_test = round(plcc(y_pred, y_test), 3)
    print('SRCC/PLCC: {}/{}'.format(SRCC_test, PLCC_test))
        
    if show_plot:
        plt.plot(y_pred, y_test, '.', markersize=0.5)
        plt.xlabel('predicted'); plt.ylabel('ground-truth'); plt.show()
    return y_test, y_pred, SRCC_test, PLCC_test

def rating_metrics(y_true, y_pred, show_plot=True):    
    """
    Print out performance measures given ground-truth (`y_true`) and predicted (`y_pred`) scalar arrays.
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    p_plcc = np.round(plcc(y_true, y_pred),3)
    p_srcc = np.round(srcc(y_true, y_pred),3)
    p_mae  = np.round(np.mean(np.abs(y_true - y_pred)),3)
    p_rmse  = np.round(np.sqrt(np.mean((y_true - y_pred)**2)),3)
    
    if show_plot:
        print('SRCC: {} | PLCC: {} | MAE: {} | RMSE: {}'.\
              format(p_srcc, p_plcc, p_mae, p_rmse))    
        plt.plot(y_true, y_pred,'.',markersize=1)
        plt.xlabel('ground-truth')
        plt.ylabel('predicted')
        plt.show()
    return (p_srcc, p_plcc, p_mae, p_rmse)

def get_train_test_sets(ids, stratify_on='MOS', test_size=(0.2, 0.2),
                        save_path=None, show_histograms=False, 
                        stratify=False, random_state=None):
    """
    Devise a train/validation/test partition for a pd.DataFrame
    Adds a column 'set' to the input `ids` that identifies each row as one of:
    ['training', 'validation', 'test'] sets.

    The partition can be stratified based on a continuous variable,
    meaning that the variable is first quantized, and then a kind of
    'class-balancing' is performed based on the quantization.

    * ids: pd.DataFrame
    * stratify_on: column name from `ids` to stratify
                   ("class balance") the partitions on
    * test_size: ratio (or number) of rows to assign to each of
                 test and validation sets respectively
                 e.g. (<validation size>, <test size>) or
                      (<validation ratio>, <test ratio>)
    * save_path: optional save path for generated partitioned table
    * show_histograms: show histograms of the distribution of the
                       stratification column
    * stratify: do stratification
    * random_state: initialize random state with a fixed value,
                    for reproducibility
    :return: modified DataFrame
    """
    if not(isinstance(test_size, tuple) or
           isinstance(test_size, list)):
        test_size = (test_size, test_size)
    ids = ids.copy().reset_index(drop=True)
    idx = list(range(len(ids)))
    if not stratify:
        strata = None
    else:
        strata = np.int32(mapmm(ids.loc[:, stratify_on], 
                                (0, stratify-1-1e-6)))
   
    idx_train_valid, idx_test = train_test_split(idx,
                                test_size=test_size[1],
                                random_state=random_state, 
                                stratify=strata)
    strata_valid = None if strata is None else strata[idx_train_valid]
    idx_train, idx_valid = train_test_split(idx_train_valid,
                           test_size=test_size[0], 
                           random_state=random_state,
                           stratify=strata_valid)

    print('Train size: {}, Validation size: {}, Test size: {}'.\
          format(len(idx_train), len(idx_valid), len(idx_test)))
    
    ids.loc[idx_train, 'set'] = 'training'
    ids.loc[idx_valid, 'set'] = 'validation'
    ids.loc[idx_test,  'set'] = 'test'
    if save_path is not None:
        ids.to_csv(save_path, index=False)
        
    if show_histograms:
        plt.hist(ids.loc[idx_train, stratify_on], density=True, 
                 facecolor='g', alpha=0.75, bins=100)
        plt.show()
        plt.hist(ids.loc[idx_valid, stratify_on], density=True, 
                 facecolor='b', alpha=0.75, bins=100)
        plt.show()
        plt.hist(ids.loc[idx_test, stratify_on], density=True, 
                 facecolor='r', alpha=0.75, bins=100)
        plt.show()

    return ids


# ---------------
# To be OBSOLETED
# ---------------

def get_model_imagenet(net_name, input_shape=None, 
                       plot=False, **kwargs):
    """Returns ImageNet models"""
    
    print('Loading model', net_name if isinstance(net_name, str)\
                                    else net_name.__name__)

    if net_name == ResNet50 or net_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape, **kwargs)
        feats = base_model.layers[-2]
        
    elif net_name == NASNetMobile or net_name == 'NASNetMobile':
        base_model = NASNetMobile(weights='imagenet',
                                  include_top=True,
                                  input_shape=input_shape, **kwargs)
        feats = base_model.layers[-3]
        
    elif net_name in list(source_module.keys()):
        base_model = net_name(weights='imagenet', include_top=False,
                              input_shape=input_shape, **kwargs)
        feats = base_model.layers[-1]
        
    else:        
        raise Exception('Unknown model ' + net_name.__name__)

    gap = GlobalAveragePooling2D(name="final_gap")(feats.output)
    model = Model(inputs=base_model.input, outputs=gap)

    if plot: plot_model(base_model, show_shapes=True,
                        to_file='plots/{}_model.png'.format(net_name.__name__))
    return model, process_input[net_name]
