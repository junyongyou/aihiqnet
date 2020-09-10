from __future__ import print_function
from __future__ import absolute_import
from builtins import map
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
import os, sys, numbers, glob, shutil, inspect, copy
import multiprocessing as mp, pandas as pd, numpy as np
from pprint import pprint
from munch import Munch
from collections import OrderedDict

import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import multi_gpu_model

from .generators import *
from .generic import *


class ModelHelper(object):
    """
    Wrapper class that simplifies default usage of Keras for training, testing, logging,
    storage of models by removing the need for some repetitive code segments (boilerplate).

    Encapsulates a Keras model and its configuration, generators for feeding the model
    during training, validation and testing, logging to TensorBoard, saving/loading
    model configurations to disk, and activations from within a model.

    When operating on a model, generators can be instantiated by the ModelHelper or pre-defined
    by the user and passed to the train/predict methods. Generators, rely on DataFrame objects
    (usually named `ids`, as they contain the IDs of data rows) for extracting data instances
    for all operations (train/validation/test).
    """
    def __init__(self, model, root_name, ids, 
                 gen_params={}, verbose=False, **params):
        """
        * model:      Keras model, compilation will be done at runtime.
        * root_name:  base name of the model, extended with
                      configuration parameters when saved
        * ids:        DataFrame table, used by generators
        * gen_params: dict for parameters for the generator
                      defaults (shuffle = True, process_fn = False,
                                deterministic = False, verbose = False)
        * verbose:    verbosity

        OTHER PARAMS
        lr   = 1e-4,               # learning rate
        loss = "MSE",              # loss function
        loss_weights   = None,     # loss weights, if multiple losses are used
        metrics        = [],       #
        class_weights  = None,     # class weights for unbalanced classes
        multiproc      = True,     # multi-processing params
        workers        = 5,        #
        max_queue_size = 10,       #
        monitor_metric      = 'val_loss',  # monitoring params
        monitor_mode        = 'min',       #       
        early_stop_patience = 20,          #       
        checkpoint_period   = 1,           #       
        save_best_only      = True,        #       
        optimizer      = optimizers.Adam(),  # optimizer object
        write_graph    = False,              # TensorBoard params
        write_images   = False,              #
        logs_root      = None,               # TensorBoard logs
        models_root    = 'models/',          # saved models path
        features_root  = 'features/',        # saved features path (by `save_activations`)
        gen_class      = None                # generator class
                                             # inferred from self.gen_params.data_path
        callbacks      = None                # option to customize callbacks list
        """
        self.model = model
        self.ids = ids
        self.verbose = verbose
        self.model_name = ShortNameBuilder(prefix=root_name+'/',
                                           sep=(':', ' '))
        self.model_cpu = None

        self.gen_params = Munch(shuffle       = True,  process_fn = False,
                                deterministic = False, verbose    = verbose, 
                                batch_size    = 32)
        self.params = Munch(lr   = 1e-4,               # learning rate
                            loss = "MSE",              # loss function
                            loss_weights   = None,     # loss weights, if multiple losses are used
                            metrics        = [],       #
                            class_weights  = None,     # class weights for unbalanced classes

                            multiproc      = True,     # multi-processing params
                            workers        = 2,        #
                            max_queue_size = 10,       #

                            monitor_metric      = 'val_loss',  # monitoring params
                            monitor_mode        = 'min',       #
                            early_stop_patience = 20,          #
                            checkpoint_period   = 1,           #
                            save_best_only      = True,        #
                            
                            optimizer      = optimizers.Adam(),  # optimizer object, its parameters
                                                                 # can changed during runtime
                            write_graph    = False,              # TensorBoard params
                            write_images   = False,              #
                            logs_root      = 'None',             # TensorBoard logs
                            models_root    = 'models/',          # saved models path
                            features_root  = 'features/',        # saved features path (by `save_activations`)
                            gen_class      = None,               # generator class inferred from self.gen_params.data_path
                            callbacks      = None                # option to customize callbacks list
                            )

        for key in list(params.keys()):
            if key not in list(self.params.keys()):
                raise Exception('Undefined parameter:' + key)

        self.gen_params.update(gen_params)        
        self.params = updated_dict(self.params, **params)

        # infer default generator class to use 
        # if params is not set yet
        if self.params.gen_class is None:
            # if has 'data_path' attribute
            if getattr(self.gen_params, 'data_path', None) is not None: 
                if self.gen_params.data_path[-3:] == '.h5':
                    self.params.gen_class = DataGeneratorHDF5
                else:
                    self.params.gen_class = DataGeneratorDisk
            else:
                raise ValueError("Cannot infer generator class")
        
        self.callbacks = None       
        self.set_model_name()

    def update_name(self, **kwargs):
        """Propagate configuration: update model_name and callbacks"""
        # update configuration parameters
        self.set_model_name()
        # update new parameters
        self.model_name(**kwargs)
        
        # always use custom callbacks, if defined
        if isinstance(self.params.callbacks, list):            
            self.callbacks = self.params.callbacks        
        else: # reset callbacks to use new model name
            self.callbacks = self.set_callbacks()
            
        return self.model_name
    
    def set_model_name(self):
        """Update model name based on parameters in self. gen_params, params and model"""
        h = self
        tostr = lambda x: str(x) if x is not None else '?'
        format_size = lambda x: '[{}]'.format(','.join(map(tostr, x)))
        loss2str = lambda x: (x if isinstance(x, (str, bytes)) else x.__name__)[:9]

        loss = h.params.loss        
        if not isinstance(loss, dict):
            loss_str = loss2str(loss)
        else:
            loss_str = '[%s]' % ','.join(map(loss2str, list(loss.values())))            
        i = '{}{}'.format(len(h.model.inputs),
                           format_size(h.gen_params.input_shape))        
        if h.model.outputs:
            o = '{}{}'.format(len(h.model.outputs),
                              format_size(h.model.outputs[0].shape[1:].as_list()))
        else: o = ''
        name = dict(i   = i,
                    o   = o,
                    l   = loss_str,
                    bsz = h.gen_params.batch_size)
        self.model_name.update(name)
        
        return self.model_name
    
    def set_callbacks(self):
        """Setup callbacks"""
        
        p = self.params
        if p.logs_root is not None:
            log_dir = os.path.join(p.logs_root, self.model_name())
            tb_callback = TensorBoard(log_dir        = log_dir,
                                      write_graph    = p.write_graph, 
                                      histogram_freq = 0,
                                      write_images   = p.write_images)        
            tb_callback.set_model(self.model)
        else:
            tb_callback = None
        
#        separator = self.model_name._ShortNameBuilder__sep[1]
        best_model_path = os.path.join(p.models_root, 
                                       self.model_name() + '_best_weights.h5')
        make_dirs(best_model_path)
        checkpointer = ModelCheckpoint(filepath = best_model_path, verbose=0,
                                       monitor  = p.monitor_metric, 
                                       mode     = p.monitor_mode, 
                                       period   = p.checkpoint_period,
                                       save_best_only    = p.save_best_only,
                                       save_weights_only = True)
        
        earlystop = EarlyStopping(monitor  = p.monitor_metric, 
                                  patience = p.early_stop_patience, 
                                  mode     = p.monitor_mode)
        
        return [earlystop, checkpointer] + ([tb_callback] if tb_callback else [])


    def _updated_gen_params(self, **kwargs):
        params = copy.deepcopy(self.gen_params)
        params.update(kwargs)
        return params
    
    def make_generator(self, ids, subset=None, **kwargs):
        """
        Create a generator of `self.params.gen_class` using new `ids`.

        * ids:    DataFrame table
        * subset: select a subset of rows based on the `set` columns
        * kwargs: updated parameters of the generator
        :return: Sequence (generator)
        """
        params = self._updated_gen_params(**kwargs)
        if subset:
            ids_ = ids[ids.set==subset]
        else:
            ids_ = ids
        return self.params.gen_class(ids_, **params)

    def test_generator(self, input_idx=0):
        """
        Basic utility to run the generator for one or more data instances in `ids`.
        Useful for testing the default generator functionality.

        * input_idx: scalar or 2-tuple of indices
        :return: generated_batch
        """
        if not isinstance(input_idx, (list, tuple)):
            ids_gen = self.ids.iloc[input_idx:input_idx+1]
        else:
            ids_gen = self.ids.iloc[input_idx[0]:input_idx[1]]

        gen = self.make_generator(ids_gen)
        x = gen[0]  # generated data
        print('First batch sizes:'); print_sizes(x)
        return x

    def set_multi_gpu(self, gpus=None):
        """
        Enable multi-GPU processing.
        Creates a copy of the CPU model and calls `multi_gpu_model`.

        * gpus: number of GPUs to use, defaults to all.
        """
        self.model_cpu = self.model
        self.model = multi_gpu_model(self.model, gpus=gpus)        

    def compile(self):
        self.model.compile(optimizer    = self.params.optimizer, 
                           loss         = self.params.loss, 
                           loss_weights = self.params.loss_weights, 
                           metrics      = self.params.metrics)
        
    def train(self, train_gen=True, valid_gen=True, 
              lr=1e-4, epochs=1, valid_in_memory=False, 
              recompile=True, verbose=True):
        """
        Run training iterations on existing model.
        Initializes `train_gen` and `valid_gen` if not defined.

        * train_gen: train generator
        * valid_gen: validation generator
        * lr:        learning rate
        * epochs:    number of epochs
        :return:     training history from self.model.fit_generator()
        """
        ids = self.ids
        print('Training model:', self.model_name())
        
        if train_gen is True:
            train_gen = self.make_generator(ids[ids.set == 'training'])
        if valid_gen is True:
            valid_gen = self.make_generator(ids[ids.set == 'validation'],
                                            shuffle       = True,
                                            deterministic = True)
        
        if recompile:
            if lr: self.params.lr = lr
            self.params.optimizer =\
                update_config(self.params.optimizer,
                              lr=self.params.lr)    
            self.compile()

        if self.verbose:
            print('\nGenerator parameters:')
            print('---------------------')
            pretty(self.gen_params)
            print('\nMain parameters:')
            print('----------------')
            pretty(self.params)
            print('\nLearning')

        if valid_in_memory:
            valid_gen.batch_size = len(valid_gen.ids)
            valid_gen.on_epoch_end()
            valid_data = valid_gen[0]
            valid_steps = 1
        else:
            valid_data = valid_gen
            if issubclass(type(valid_gen), 
                          keras.utils.Sequence):
                valid_steps = len(valid_gen)
            else:
                valid_steps = 1
        
        # always use custom callbacks, if defined
        if isinstance(self.params.callbacks, list):            
            self.callbacks = self.params.callbacks
        # set callbacks if first run, otherwise do not reset
        elif self.callbacks is None:
            self.callbacks = self.set_callbacks()
                            
        if issubclass(type(train_gen), 
                      keras.utils.Sequence): 
            # train using the generator            
            history = self.model.fit_generator(
                             train_gen, epochs = epochs,
                             steps_per_epoch   = len(train_gen),
                             validation_data   = valid_data, 
                             validation_steps  = valid_steps,
                             workers           = self.params.workers, 
                             callbacks         = self.callbacks,
                             max_queue_size    = self.params.max_queue_size,
                             class_weight      = self.params.class_weights,
                             use_multiprocessing = self.params.multiproc,
                             verbose             = verbose)
        else:
            # training data is passed in train_gen
            X, y = train_gen            
            steps_per_epoch = X.shape[0]//self.gen_params.batch_size
            
            history = self.model.fit(X, y,
                             batch_size      = self.gen_params.batch_size,
                             epochs          = epochs,
                             callbacks       = self.callbacks,                
                             validation_data = valid_data,
                             shuffle         = self.gen_params.shuffle,
                             class_weight    = self.params.class_weights,
                             verbose         = verbose)

        return history

    def clean_outputs(self, force=False):
        """
        Delete training logs or models created by the current helper configuration.
        Identifies the logs by the configuration paths and `self.model_name`.
        Asks for user confirmation before deleting any files.
        """
        log_dir = os.path.join(self.params.logs_root, self.model_name())
        model_path = os.path.join(self.params.models_root, self.model_name()) + '*.h5'
        model_path = model_path.replace('[', '[[]')
        model_files = glob.glob(model_path)

        if os.path.exists(log_dir):
            print('Found logs:')
            print(log_dir)
            if force or raw_confirm('Delete?'):
                print('Deleting', log_dir)
                shutil.rmtree(log_dir)
        else:
            print('(No logs found)')

        if model_files:
            print('Found model(s):')
            print(model_files)
            if force or raw_confirm('Delete?'):
                for mf in model_files: 
                    print('Deleting', mf)
                    os.unlink(mf)
        else:
            print('(No models found)')
        
    def predict(self, test_gen=None, output_layer=None, 
                repeats=1, batch_size=None, remodel=True):
        """
        Predict on `test_gen`.

        * test_gen:     generator used for prediction
        * output_layer: layer at which activations are computed (defaults to output)
        * repeats:      how many times the prediction is repeated (with different augmentation)
        * batch_size:   size of each batch
        * remodel:      if `true` then change model such that new output is `output_layer`
        :return:        if `repeats` == 1 then `np.ndarray` else `list[np.ndarray]`
        """
        if not test_gen:
            params_test = self._updated_gen_params(shuffle       = False, 
                                                   fixed_batches = False)
            if batch_size: params_test.batch_size = batch_size
            test_gen = self.params.gen_class(self.ids[self.ids.set == 'test'],
                                             **params_test)
        if output_layer is not None and remodel:
                # get last partial-matching layer
                layer_name = [l.name for l in self.model.layers 
                              if output_layer in l.name][-1]
                output_layer = self.model.get_layer(layer_name)
                print('Output of layer:', output_layer.name)
                if isinstance(output_layer, Model):
                    outputs = output_layer.outputs[0]
                else:
                    outputs = output_layer.output
                print('Output tensor:', outputs)
                model = Model(inputs  = self.model.input, 
                              outputs = outputs)
        else: 
            model = self.model

        preds = []
        for i in range(repeats):
            y_pred = model.predict_generator(test_gen, workers=1, verbose=0,
                                             use_multiprocessing=False)
            if not remodel and output_layer is not None:
                y_pred = dict(list(zip(model.output_names, y_pred)))[output_layer]
            preds.append(y_pred)
        return preds[0] if repeats == 1 else preds
    
    def validate(self, valid_gen=True, verbose=2, 
                 batch_size=32, recompile=True):
        if valid_gen is True:
            ids = self.ids
            valid_gen = self.make_generator(ids[ids.set == 'validation'],
                                            shuffle       = True,
                                            deterministic = True)        
        print('Validating performance')
        if recompile: self.compile()

        if issubclass(type(valid_gen), keras.utils.Sequence): 
            r = self.model.evaluate_generator(valid_gen,
                                              verbose=verbose)
        else:
            X_valid, y_valid = valid_gen
            r = self.model.evaluate(X_valid, y_valid, 
                                    batch_size=batch_size, 
                                    verbose=verbose)
        perf_metrics = dict(list(zip(self.model.metrics_names, 
                                     force_list(r))))
        if verbose==2:
            pretty(OrderedDict((k, perf_metrics[k]) 
                               for k in sorted(perf_metrics.keys())))
        return perf_metrics 

    def set_trainable(self, index):
        """
        Convenience method to set trainable layers.
        Layers up to `index` are frozen; the remaining
        after `index` are set as trainable.
        """
        for layer in self.model.layers[:index]:
            layer.trainable = False
        for layer in self.model.layers[index:]:
            layer.trainable = True

    def load_model(self, model_name='', best=True, 
                   from_weights=True, by_name=False, verbose=1):
        """
        Load model from file.

        * model_name:   new model name, otherwise self.model_name()
        * best:         load the best model, or otherwise final model
        * from_weights: from weights, or from full saved model
        * by_name:      load layers by name
        :return:        true if model was loaded successfully, otherwise false
        """
        model_name = model_name or self.model_name()
        model_file_name = (model_name + ('_best' if best else '_final') + 
                          ('_weights' if from_weights else '') + '.h5')
        model_path = os.path.join(self.params.models_root, model_file_name)
        if not os.path.exists(model_path):
            if verbose:
                print('Model NOT loaded:', model_file_name, 'does not exist')
            return False
        else:
            if from_weights:
                self.model.load_weights(model_path, by_name=by_name)
                if verbose:
                    print('Model weights loaded:', model_file_name)
            else:
                self.model = load_model(model_path)
                if verbose:
                    print('Model loaded:', model_file_name)
            return True

    def save_model(self, weights_only=False, model=None, 
                   name_extras='', best=False, verbose=1):
        """
        Save model to HDF5 file.

        * weights_only: save only weights,
                        or full model otherwise
        * model:        specify a particular model instance,
                        otherwise self.model is saved
        * name_extras:  append this to model_name
        * best:         save as best model, otherwise final model
        """
        model = model or self.model
        if verbose:
            print('Saving model', model.name, 'spanning',\
                  len(self.model.layers), 'layers')
        model_type = 'best' if best else 'final'
        if weights_only:
            model_file = self.model_name() + name_extras + '_' + model_type + '_weights.h5'
            model.save_weights(os.path.join(self.params.models_root, model_file))
            if verbose:
                print('Model weights saved:', model_file)
        else:
            model_file = self.model_name() + name_extras + '_' + model_type + '.h5'
            model.compile(optimizer=self.params.optimizer, loss="mean_absolute_error")
            model.save(os.path.join(self.params.models_root, model_file))
            if verbose:
                print('Model saved:', model_file)

    def save_activations(self, output_layer=None, file_path=None, ids=None,
                         groups=1, verbose=False, overwrite=False, name_suffix='',
                         save_as_type=np.float32, postprocess_fn=None):
        """
        Save activations from a particular `output_layer` to an HDF5 file.

        * output_layer:   if not None, layer name, otherwise use the model output
        * file_path:      HDF5 file path
        * ids:            data entries to compute the activations for, defaults to `self.ids`
        * groups:         a number denoting the number of augmentation repetitions, or list of group names
        * verbose:        verbosity
        * overwrite:      overwrite HDF5 file
        * name_suffix:    append suffix to name of file (before ext)
        * save_as_type:   save as a different data type, defaults to np.float32
        * postprocess_fn: post-processing function to apply to the activations
        """
        if ids is None:
            ids = self.ids
        if isinstance(groups, numbers.Number):
            groups_count = groups
            groups_list = list(map(str, list(range(groups))))
        else:
            groups_count = len(groups)
            groups_list  = list(map(str, groups))

        if file_path is None:         
            short_name = self.model_name.subset(['i', 'o'])
            short_name(grp = groups_count,
                       lay = (output_layer or 'final'))
            file_path = os.path.join(self.params.features_root, 
                                     str(short_name) + name_suffix + '.h5')
            make_dirs(file_path)

        params = self._updated_gen_params(shuffle       = False, 
                                          verbose       = verbose,
                                          fixed_batches = False,
                                          group_by      = None)
        if verbose>1:
            print('Saving activations for layer:', (output_layer or 'final'))
            print('file:', file_path)

        data_gen = self.make_generator(ids, **params)

        for group_name in groups_list:
            activ = self.predict(data_gen, 
                                 output_layer = output_layer)
            activ = activ.astype(save_as_type)
            if len(activ.shape)==1:
                activ = np.expand_dims(activ, 0)            
            if postprocess_fn:
                activ = postprocess_fn(activ)
                
            with H5Helper(file_path, 
                          overwrite = overwrite, 
                          verbose   = verbose) as h:
                if groups == 1:
                    h.write_data(activ, list(ids.loc[:,data_gen.inputs[0]]))
                else:
                    h.write_data([activ], list(ids.loc[:,data_gen.inputs[0]]), 
                                 group_names=[group_name])
            del activ

    def save_features(self, process_gen, batch_size = 1024,
                      save_as_type = np.float32, overwrite = False):
        """
        Save augmented features to HDF5 file.
        
        * process_gen: if function: applies `process_gen` as `self.gen_params.preprocess_fn`
                       if generator: defines `preprocess_fn` functions to use for each augmentation
        * batch_size: batch size used for storing activations (in an `np.ndarray`)
        * save_as_type: convert the activations to this type, to reduce required storage
        * overwrite: overwrite features file
        """
        ids = self.ids
        print('[Saving features]')

        if not inspect.isgeneratorfunction(process_gen):
            process_gen = [(process_gen,None)]
        else:
            process_gen = process_gen()
        
        first_verbose = 2
        for (process_fn, args) in process_gen:
            self.gen_params.process_fn = process_fn
            if args:
                if isinstance(args, dict):
                    arg_str = ', '.join(['{}:{}'.format(*a) for a in args.items()])
                else: 
                    arg_str = str(args)
                print('Augmentation args "' + arg_str + '"')

            numel = len(ids)
            for i in range(0,numel,batch_size):
                istop = min(i+batch_size, numel)
                print('\nImages',i,':',istop)
                ids_batch = ids.iloc[i:istop]

                self.save_activations(ids=ids_batch, verbose = first_verbose, 
                                      groups       = [arg_str] if args else 1,
                                      save_as_type = save_as_type,
                                      overwrite    = overwrite)
                overwrite = False
                first_verbose = 1

class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, valid_gen, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.valid_gen = valid_gen  # The validation generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        X, y = self.valid_gen[0]
        X = np.float32(X);
        y = np.float32(y)
        sample_weights = np.ones(X.shape[0], dtype=np.float32)
        self.validation_data = [X, y, sample_weights, np.float32(0.0)]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


# MISC helper functions

def get_layer_index(model, name):
    """Get index of layer by name"""
    for idx, layer in enumerate(model.layers):
        if layer.name == name:
            return idx

def get_activations(im, model, layer):
    """Get activations from `layer` in `model` using `K.function`"""
    if len(im.shape) < 4:
        im = np.expand_dims(im, 0)
    inputs = [K.learning_phase()] + model.inputs
    fn = K.function(inputs, [layer.output])
    act = fn([0] + [im])
    act = np.squeeze(act)
    return act
