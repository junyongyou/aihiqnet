from __future__ import print_function
from __future__ import absolute_import

# ignore warnings scikit-image, numpy > 1.17 in tensorflow
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from . import generic, tensor_ops, image_utils, generators, model_helper, applications

# remove tensorflow warnings
import logging
class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning1 = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg        
        tf_warning2 = 'is deprecated' in msg
        return not (tf_warning1 or tf_warning2)
logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

print('Loaded KU')