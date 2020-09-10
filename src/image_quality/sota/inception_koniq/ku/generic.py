from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np, pandas as pd
import multiprocessing as mp
import os, scipy, h5py, time, sys
import urllib, shutil, subprocess
from munch import Munch
from sklearn.model_selection import train_test_split

if sys.version_info.major == 2:
    input = raw_input

# Helps with the DataGeneratorHDF5
class H5Helper(object):
    """
    Read/Write named data sets from/to an HDF5 file.
    The structure of the data inside the HDF5 file is:
    'group_name/dataset_name' or 'dataset_name'

    Enables reading from random groups e.g. 'augmentation_type/file_name'
    such that the helper can be used during training Keras models.
    """
    def __init__(self, file_name, file_mode=None, 
                 memory_mapped=False, overwrite=False, 
                 backing_store=False, verbose=False):
        """
        * file_name: HDF5 file path
        * file_mode: one of 'a','w','r','w+','r+'
        * memory_mapped: enables memory mapped backing store
        * overwrite: over-write existing file
        * backing_store: use another backing store
        * verbose: verbosity
        """
        self.hf = None
        self.file_name = file_name
        self.verbose = verbose
        self.memory_mapped = memory_mapped
        self.backing_store = backing_store
        self._lock = mp.Lock()
        _file_mode = file_mode or ('w' if overwrite else 'a')
        with self._lock:
            if memory_mapped:
                # memory mapping via h5py built-ins
                self.hf = h5py.File(file_name, _file_mode, driver='core', 
                                    backing_store = backing_store)
            else:
                self.hf = h5py.File(file_name, _file_mode)

    def _write_datasets(self, writer, data, dataset_names):
        # internal to the class
        for i in range(len(data)):
            if self.verbose:
                show_progress(i, len(data), prefix='Writing datasets')
            writer.create_dataset(dataset_names[i], data=data[i, ...])

    def write_data(self, data, dataset_names, group_names=None):
        """
        Write `data` to HDF5 file, using `dataset_names` for datasets,
        and optionally `group_names` for groups.

        * data:          if `group_names` is None: np.ndarray of N data instances of size N x [...]
                         else list of np.ndarray of N data instances of size N x [...] each
        * dataset_names: list of strings
        * group_names:   None, or list of strings
        """
        with self._lock:            
            assert isinstance(dataset_names, list), "`dataset_names` is of type {} and should be `list`".format(type(dataset_names))
                   
            hf = self.hf
            if group_names is None:
                assert not isinstance(data, list),\
                       '`data` should be a `numpy.ndarray` when no groups are specified'
                self._write_datasets(hf, data, dataset_names)
            else:
                assert isinstance(data, list) and len(data) == len(group_names),\
                       'Each group name should correspond to a `data` list entry, `len(data)`={}, while `len(grou_names)`={}'.\
                       format(len(data), len(group_names))
                for i, name in enumerate(group_names):
                    group = hf.require_group(name)
                    self._write_datasets(group, data[i], dataset_names)

    def _read_datasets(self, reader, dataset_names):
        # internal to the class
        name = dataset_names[0]
        data0 = reader[name][...]
        data = np.empty((len(dataset_names),) + data0.shape, 
                        dtype=data0.dtype)
        data[0,...] = data0
        for i in range(1, len(dataset_names)):
            if self.verbose:
                show_progress(i, len(data), prefix='Reading datasets')
            data[i, ...] = reader[dataset_names[i]][...]
        return data
  
    def read_data(self, dataset_names, group_names=None):
        """
        Read `dataset_names` from HDF5 file, optionally using `group_names`.

        * dataset_names: list of strings
        * group_names:   None, or list of strings
        :return:         np.ndarray
        """
        with self._lock:
            hf = self.hf
            assert isinstance(dataset_names, list), "`dataset_names` is of type {} and should be `list`".format(type(dataset_names))
            if group_names is None:
                return self._read_datasets(hf, dataset_names)
            else:
                return [self._read_datasets(hf[group_name], dataset_names)
                        for group_name in group_names]        

    def read_data_random_group(self, dataset_names):
        """
        Reads `dataset_names` each one from a random group.
        At least one group must exist.

        * dataset_names: list of strings
        :return: np.ndarray
        """
        with self._lock:
            hf = self.hf
            group_names = np.array(self.group_names)
            idx = np.random.randint(0, len(group_names), len(dataset_names))
            names = ['{}/{}'.format(g, d) for g, d in zip(group_names[idx], dataset_names)]
            return self._read_datasets(hf, names)

    def summary(self, print_limit=100):
        """
        Prints a summary of the contents of the HDF5 file.
        Lists all groups and first `print_limit` datasets for each group.

        * print_limit: number of datasets to list per group.
        """
        size_bytes = os.path.getsize(self.file_name)
        print('File size: {:.3f} GB'.format(size_bytes/(2.0**30)))
        print('Groups:')

        hf = self.hf
        keys = list(hf.keys())
        for i, group_name in enumerate(keys):
            if i > print_limit:
                print('[...] showing %d of %d' % (print_limit, len(keys)))
                break
            print('*',group_name, '\b/')
            group = hf[group_name]
            try:
                group_keys = list(group.keys())
                print(' ', end=' ')
                for j, dataset_name in enumerate(group_keys):
                    if j > print_limit:
                        print('[...] showing %d of %d' % (print_limit, len(group_keys)))
                        break
                    print(dataset_name, end=' ')
                if len(group_keys):
                    print()
            except: pass

    @property
    def group_names(self):
        """
        :return: list of group names
        """
        return list(self.hf.keys())

    @property
    def dataset_names(self):
        """
        :return: list of dataset names 
                 (if groups are present, from the first group)
        """
        values = list(self.hf.values())
        if isinstance(values[0], h5py._hl.dataset.Dataset):
            return list(self.hf.keys())
        else:
            return list(values[0].keys())

    # enable 'with' statements
    def __del__(self):
        self.__exit__(None, None, None)

    def __enter__(self): 
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hf is not None:
            self.hf.flush()
            self.hf.close()
            del self.hf
            self.hf = None
                

def minmax(x):
    """
    Range of x.

    * x: list or np.ndarray
    :return: (min, max)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x.min(), x.max()

def mapmm(x, new_range = (0, 1)):
    """
    Remap values in `x` to `new_range`.

    * x: np.ndarray
    * new_range: (min, max)
    :return: np.ndarray with values mapped to [new_range[0], new_range[1]]
    """
    mina, maxa = new_range
    if not type(x) == np.ndarray: 
        x = np.asfarray(x, dtype='float32')
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    minx, maxx = minmax(x)
    if minx < maxx:
        x = old_div((x-minx),(maxx-minx))*(maxa-mina)+mina
    return x

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    return scipy.stats.pearsonr(x, y)[0]

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()    
    yranks = pd.Series(ys).rank()    
    return plcc(xranks, yranks)
srcc = srocc

def dist2mos(x, scale=np.arange(1, 6)):
    """
    Find the MOS of a distribution of scores `x`, given a `scale`.
    e.g. x=[0,2,2], scale=[1,2,3] => MOS=2.5
    """
    x = old_div(x, np.reshape(np.sum(x*1., axis=1), (len(x), 1)))
    return np.sum(x * scale, axis=1)
    
def force_tuple(x):
    """Make tuple out of `x` if not already a tuple or `x` is None"""
    if x is not None and not isinstance(x, tuple):
        return (x,)
    else:
        return x
    
def force_list(x):
    """Make list out of `x` if not already a list or `x` is None"""
    if x is not None and not isinstance(x, list):
        return [x]
    else:
        return x

def make_dirs(filename):
    """
    Create directory structure described by `filename`.
    * filename: a valid system path
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except: pass
        
def updated_dict(d, only_existing=True, **updates):
    """
    Update dictionary `d` with `updates`, optionally changing `only_existing` keys.

    * d: dict
    * only_existing: do not add new keys, update existing ones only
    * updates: dict
    :return: updated dictionary
    """
    d = d.copy()
    if only_existing:
        common = {key: value for key, value in list(updates.items())
                  if key in list(d.keys())}     
        d.update(common)
    else:
        d.update(updates)
    return d

def chunks(l, n):
    """Yields successive `n`-sized chunks from list `l`."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def pretty_print(d, indent=0, key_sep=':', trim=True):
    """
    Pretty print dictionary, recursively.

    * d: dict
    * indent: indentation amount
    * key_sep: separator printed between key and values
    * trim: remove redundant white space from printed values
    """
    if indent == 0 and not isinstance(d, dict):
        d = d.__dict__
    max_key_len = 0
    keys = list(d.keys())
    if isinstance(keys, list) and len(keys)>0:
        max_key_len = max([len(str(k)) for k in keys])        
    for key, value in list(d.items()):
        equal_offset = ' '*(max_key_len - len(str(key)))
        print('\t' * indent + str(key) + key_sep, end=' ')
        if isinstance(value, dict):
            print()
            pretty(value, indent+1)
        else:
            value_str = str(value).strip()
            if trim:
                value_str = ' '.join(value_str.split())
            if len(value_str) > 70:
                value_str = value_str[:70] + ' [...]' 
            print(equal_offset + value_str)
            
pretty = pretty_print

class ShortNameBuilder(Munch):
    """
    Utility for building short (file) names
    that contain multiple parameters.

    For internal use in ModelHelper.
    """
    def __init__(self, prefix='', sep=('', '_'), 
                 max_len=32, **kwargs):
        self.__prefix  = prefix
        self.__sep     = sep
        self.__max_len = max_len
        super(ShortNameBuilder, self).__init__(**kwargs)

    def subset(self, selected_keys):
        subself = ShortNameBuilder(**self)
        for k in list(subself.keys()):
            if k not in selected_keys and \
               '_ShortNameBuilder' not in k:
                del subself[k]
        return subself

    def __call__(self, subset=None, **kwargs):
        self.update(**kwargs)
        return str(self)
    
    def __str__(self):
        def combine(k, v):
            k = str(k)[:self.__max_len]
            v = str(v)[:self.__max_len]
            return (k + self.__sep[0] + v if v else k)
        return self.__prefix + \
               self.__sep[1].join([combine(k, self[k])
                    for k in sorted(self.keys())
                    if '_ShortNameBuilder' not in k])
    
def check_keys_exist(new, old):
    """
    Check that keys in `new` dict existing in `old` dict.
    * new: dict
    * old: dict
    :return: exception if `new` keys don't existing in `old` ones

    Utility function used internally.
    """
    for key in list(new.keys()):
        if key not in list(old.keys()):
            raise Exception('Undefined parameter: "%s"' % key)
            
def get_sizes(x, array_marker='array'):
    """
    String representation of the shapes of arrays in lists/tuples.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        content = ', '.join([get_sizes(_x_, array_marker) for _x_ in x])
        if isinstance(x, list):
            return '[' + content + ']'
        else:
            return '(' + content + ')'
    elif hasattr(x, 'shape'):
        return array_marker + '<' + ','.join(map(str, x.shape)) + '>'
    else: return '<1>'
            
def print_sizes(x, array_marker=''):
    """
    Prints get_sizes(x)
    """
    print(get_sizes(x, array_marker=array_marker))
        
def raw_confirm(message):
    """
    Ask for confirmation.
    
    * message: message to show
    :return: true if confirmation given, false otherwise
    """
    confirmation = input(message + " (y/[n])")
    if not confirmation:
        return False  # do not confirm by default
    else:
        return confirmation.lower()[0] == "y"

def update_config(obj, **kwargs):
    """
    Update configuration of Keras `obj` e.g. layer, model.

    * obj: object that has .get_config() and .from_config() methods.
    * kwargs: dict of updates
    :return: updated object
    """
    cfg = obj.get_config()
    cfg.update(**kwargs)
    return obj.from_config(cfg)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(('[%s]' % self.name), end=' ')
        print('elapsed: %s seconds' % round(time.time() - self.tstart, 4))

        
def partition_rows(t, test_size=0.2, set_name='set', 
                   set_values=['training', 'test'], 
                   random_state=None, copy=True):
    if copy: t = t.copy()
    t = t.reset_index(drop=True)
    itrain, itest = train_test_split(list(range(len(t))),
                                     test_size=test_size,
                                     random_state=random_state)
    t.loc[itrain, set_name] = set_values[0]
    t.loc[itest,  set_name] = set_values[1]
    return t

def array_overlap(a, b):
    """
    Returns indices of overlapping values in two arrays.
    From: https://www.followthesheep.com/?p=1366
    """ 
    ia=np.argsort(a)
    ib=np.argsort(b)
    
    # use searchsorted:
    sort_left_a  = a[ia].searchsorted(b[ib], side='left')
    sort_right_a = a[ia].searchsorted(b[ib], side='right')
    #
    sort_left_b  = b[ib].searchsorted(a[ia], side='left')
    sort_right_b = b[ib].searchsorted(a[ia], side='right')

    # which values of b are also in a?
    inds_b = (sort_right_a-sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a = (sort_right_b-sort_left_b > 0).nonzero()[0]

    return ia[inds_a], ib[inds_b]


# Print iterations progress
# Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def show_progress(iteration, total, prefix = '', suffix = '', decimals = 0, 
                  length = 50, fill = '=', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    iteration_ = iteration+1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration_ / float(total)))
    filledLength = int(length * iteration_ / total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration_ == total: 
        print()

def download_archive(data_url, data_root, cache_root=None, 
                     delete_archive=True, verbose=False, archive_type=None):
    archive_name = data_url.split('/')[-1]
    data_path = os.path.join(data_root, archive_name)
    make_dirs(data_root)

    if cache_root:
        make_dirs(cache_root)
        cache_path = os.path.join(cache_root, archive_name)
        if not os.path.exists(cache_path):
            if verbose:
                print('downloading {} via cache {}'.format(data_url, cache_root))
            urllib.request.urlretrieve(data_url, cache_path)
        if verbose:
            print('copying {} to {}'.format(archive_name, data_root))
        shutil.copy(cache_path, data_root)
    else:
        if not os.path.exists(data_path):
            if verbose:
                print('downloading {} to {}'.format(data_url, data_path))            
            urllib.request.urlretrieve(data_url, data_path)

    if verbose:
        print('unpacking archive {} to {}'.format(archive_name, data_root))
       
    if  archive_type is None:
        archive_type = archive_name.split('.')[-1]
        
    if archive_type == 'zip':
        failed_command = subprocess.run(["unzip","-qq","-o",data_path,"-d",data_root], check=True)
    else:
        failed_command = subprocess.run(["tar","-xf",data_path,"-C",data_root], check=True)
        
    if failed_command.returncode:
        print("unpack failed: %d" % failed_command.returncode)
        
    if delete_archive:
        os.unlink(data_path)