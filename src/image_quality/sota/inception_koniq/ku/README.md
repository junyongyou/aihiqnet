# Keras Utilities (KU)

The project contains utilities for image assessment development with Keras/Tensorflow, including utilities for model training, custom generators, image management and augmentation. This is an extension of [kutils](https://github.com/subpic/kutils).

## Overview

Some of the key components of each file:

**`model_helper.py`**:

* `ModelHelper`: Wrapper class that simplifies default usage of Keras for regression models.

**`generators.py`**:

* `DataGeneratorDisk`, `DataGeneratorHDF5`: Keras generators for on-disk images, and HDF5 stored features/images

**`image_utils.py`**:

* various utility functions for manipulating images (read, write to HDF5, batch resize, view batch)

**`image_augmenter.py`**:

* `ImageAugmenter`: Create custom image augmentation functions for training Keras models.

**`generic.py`**:

* `H5Helper`: Manage named data sets in HDF5 files, for us in Keras generators.
* `ShortNameBuilder`: Utility for building short (file) names that contain multiple parameters.

**`applications.py`**:

* `model_inception_multigap`, `model_inceptionresnet_multigap`: Model definitions for extracting MLSP narrow features
* `model_inception_pooled`, `model_inceptionresnet_pooled`: Model definitions for extracting MLSP wide features

You can find more information in the docstrings.
