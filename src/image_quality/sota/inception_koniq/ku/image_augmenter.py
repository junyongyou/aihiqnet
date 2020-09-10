from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import math, os, numpy as np, glob
import scipy.ndimage.interpolation
import skimage.transform as transform
from numpy import interp
from numpy.random import rand
import PIL, random
from PIL import Image
import matplotlib.pyplot as plt
from .generic import *

class ImageAugmenter(object):
    """
    Provides methods to easily transform images.
    Meant for creating custom image augmentation functions for training Keras models.
    e.g. # randomly crop and flip left-right a 224x224 patch out of an image
         process_fn = lambda im: ImageAugmenter(im).crop((224,224)).fliplr().result
         # process_fn can be passed as an argument to keras.utils.Sequence objects (data generators)

    Provides various pre-defined customizable transformations, all randomizable:
    rotate, crop, fliplr, rescale, resize. The transformations can be easily chained.
    """
    def __init__(self, image, remap=False, verbose=False):
        """
        * image: image to be transformed, np.ndarray
        * remap: remap values to [0,1] for easier to apply transformations
                 these are mapped back to the initial range when .result is called
        * verbose: enable verbose prints
        """
        self._rotation_angle = 0
        self._original_range = minmax(image)
        self._remap = remap
        self.image = image if not self._remap else mapmm(image)
        self.verbose = verbose
        
    def rotate(self, angle, random=True):
        """
        Rotate self.image

        * angle: if `random` then rotation angle is a random value between [-`angle`, `angle`]
                 otherwise rotation angle is `angle`
        * random: random or by a fixed amount
        :return: self
        """
        if angle != 0 and random:
            # rotation angle is random between [-angle, angle]
            self._rotation_angle += (rand(1)-0.5)*2*angle
        else:
            self._rotation_angle += angle
            
        self.image = transform.rotate(self.image, self._rotation_angle, 
                                      resize=False, cval=0, 
                                      clip=True, preserve_range=True, 
                                      mode='symmetric')            
        return self
    
    def crop(self, crop_size, crop_pos=None, clip_rotation=False):
        """
        Crop a patch out of self.image. Relies on `extract_patch`.

        * crop_size: dimensions of the crop (pair of H x W)
        * crop_pos: if None, then a random crop is taken, otherwise the given `crop_pos` position is used
                    pair of relative coordinates: (0,0) = upper left corner, (1,1) = lower right corner
        * clip_rotation: clip a border around the image, such that the edge resulting from
                         having rotated the image is not visible
        :return: self
        """
        # equally crop in both dimensions if only one number is provided
        if not isinstance(crop_size, (list, tuple)):
            crop_size = [crop_size, crop_size]
        # if using a ratio crop, compute actual crop size
        crop_size = [np.int32(c*dim) if 0 < c <= (1+1e-6) else c\
                     for c, dim in zip(crop_size, self.image.shape[:2])]
        
        if self.verbose:
            print('image_size:', self.image.shape, 'crop_size:', crop_size)

        if crop_pos is None:
            if crop_size != self.image.shape[:2]:
                if clip_rotation:
                    lrr = largest_rotated_rect(self.image.shape[0], 
                                               self.image.shape[1], 
                                               math.radians(self._rotation_angle))
                    x, y = self.image.shape, lrr
                    border = (old_div((x[0]-y[0]),2), old_div((x[1]-y[1]),2))
                else:
                    border = (0, 0)
                self.image = extract_random_patch(self.image,
                                                  patch_size = crop_size, 
                                                  border     = border)
        else:
            if crop_size != self.image.shape[:2]:
                self.image = extract_patch(self.image, 
                                           patch_size     = crop_size, 
                                           patch_position = crop_pos)
        return self

    def cropout(self, crop_size, crop_pos=None, fill_val=0):
        """
        Cropout a patch of self.image and replace it with `fill_val`. Relies on `cropout_patch`.
        
        * crop_size: dimensions of the cropout (pair of H x W)
        * crop_pos:  if None, then a random cropout is taken, otherwise the given `crop_pos` position is used
                     pair of relative coordinates: (0,0) = upper left corner, (1,1) = lower right corner
        * fill_val:  value to fill the cropout with
        :return:     self
        """
        # size of the cropout is equal in both dimensions if only one number is provided
        if not isinstance(crop_size, (list, tuple)):
            crop_size = [crop_size, crop_size]
        # if using a ratio cropout, compute actual cropout size
        crop_size = [np.int32(c*dim) if isinstance(c, float) and (0 < c) and (c <= 1.) else c\
                     for c, dim in zip(crop_size, self.image.shape[:2])]
             
        if self.verbose:
            print('image_size:', self.image.shape, 'crop_size:', crop_size, 'fill_val:', fill_val)

        if crop_pos is None:
            if crop_size != self.image.shape[:2]:
                border = (0, 0)
                self.image = cropout_random_patch(self.image,
                                                  patch_size = crop_size,
                                                  border     = border,
                                                  fill_val   = fill_val)
        else:
            if crop_size != self.image.shape[:2]:
                self.image = cropout_patch(self.image,
                                           patch_size     = crop_size,
                                           patch_position = crop_pos,
                                           fill_val       = fill_val)
        return self
    
    def fliplr(self, do=None):
        """
        Flip left-right self.image

        * do: if None, random flip, otherwise flip if do=True
        :return: self
        """
        if (do is None and rand(1) > 0.5) or do:
            self._rotation_angle = -self._rotation_angle
            self.image = np.fliplr(self.image)
        return self
    
    def rescale(self, target, proportion=1, min_dim=False):
        """
        Rescale self.image proportionally

        * target: (int) target resolution relative to the reference image resolution
                  taken to be either the height if `min_dim` else min(height, width)
                  (float) zoom level
        * proportion: modulating factor for the zoom
                      when proportion=1 target zoom is unchanged
                      when proportion=0 target zoom=1 (original size)
        * min_dim: bool
        :return: self
        """
        if isinstance(target, int): # target dimensions
            if not min_dim:
                # choose height for zoom
                zoom_target = self.image.shape[0] 
            else:
                # choose minimum dimension
                zoom_target = min(self.image.shape[0],
                                  self.image.shape[1])
            zoom = old_div(1. * target, zoom_target)
        else:
            zoom = target
        zoom = (1-proportion) + proportion*zoom
            
        self.image = transform.rescale(self.image, zoom, 
                                       preserve_range=True,
                                       mode='reflect')
        return self
    
    def resize(self, size, ensure_min=False, fit_frame=False):
        """
        Resize image to target dimensions, exact or fitting inside frame

        * size: (height, width) tuple
        * ensure_min: if true, `size` is the minimum size allowed
                      a dimension is not changed unless it is below the minimum size
        * fit_frame: size concerns the dimensions of the frame that the image is to be 
                     fitted into, while preserving its aspect ratio
        :return: self
        """
        imsz = self.image.shape[:2] # (height, width)
                
        if not fit_frame:
            # resize if needed only
            if (not ensure_min and size != imsz) or\
               (ensure_min and (imsz[0] < size[0] or imsz[1] < size[1])):
                if ensure_min:
                    size = [max(a, b) for a, b in zip(imsz, size)]
                self.image = transform.resize(self.image, size, 
                                              preserve_range=True)
        else:
            image_height, image_width = imsz
            frame_height, frame_width = size
            aspect_image = float(image_width)/image_height
            aspect_frame = float(frame_width)/frame_height
            if aspect_image > aspect_frame: # fit width
                target_width = frame_width
                target_height = frame_width / aspect_image
            else: # fit height
                target_height = frame_height
                target_width = frame_height * aspect_image

            target_width, target_height = int(np.round(target_width)), int(np.round(target_height))

            self.image = transform.resize(self.image, (target_height, target_width), 
                                          preserve_range=True)
        return self

    @property
    def result_image(self):
        return array_to_img(self.result)

    @property
    def result(self):
        """
        :return: transformed image
        """
        if self._remap:
            return mapmm(self.image, self._original_range)
        else:
            return self.image

        
# -- utility functions --
            
def get_patch_dims(im, patch_size, patch_position):
    """
    Returns the dimensions of an image patch of size `patch_size`,
    with its center at `patch_position` expressed as a ratio of the image's H and W
    
    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    :return:          tuple of (upper left corner X coordinate, 
                                upper left corner Y coordinate,
                                lower right corner X coordinate,
                                lower right corner Y coordinate)
    """
    Py, Px         = patch_position
    H, W, _        = im.shape
    H_crop, W_crop = patch_size
    
    H_crop, W_crop = min(H, H_crop), min(W, W_crop)
    Y_max, X_max   = (H - H_crop, W - W_crop)
    Yc, Xc         = H*Py, W*Px

    X0, Y0 = Xc-old_div(W_crop,2), Yc-old_div(H_crop,2)
    X0, Y0 = min(max(int(X0), 0), X_max),\
             min(max(int(Y0), 0), Y_max)
    
    return (X0, Y0, X0+W_crop, Y0+H_crop)

def extract_patch(im, patch_size=(224, 224), 
                  patch_position=(0.5, 0.5)):
    """
    Extract a patch of size `patch_size`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    :return:          np.ndarray
    """
    (X0, Y0, X1, Y1) = get_patch_dims(im, patch_size, patch_position)
    return im[Y0:Y1, X0:X1, ]

def get_random_patch_dims(im, patch_size, border):
    """
    Returns the dimensions of a random image patch of size `patch_size`,
    with the center of the patch inside `border`
    
    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
    :return:      tuple of (upper left corner X coordinate, 
                            upper left corner Y coordinate,
                            lower right corner X coordinate,
                            lower right corner Y coordinate)
    """
    H, W, _        = im.shape
    H_crop, W_crop = patch_size
    
    H_crop, W_crop = min(H, H_crop), min(W, W_crop)    
    Y_min, X_min   = border
    Y_max, X_max   = (H - H_crop - Y_min, W - W_crop - X_min)
    
    if Y_max < Y_min: 
        Y_min = old_div((H - H_crop), 2)
        Y_max = Y_min
    
    if X_max < X_min:
        X_min = old_div((W - W_crop), 2)
        X_max = X_min
    
    Y0 = int(np.round(rand(1)*(Y_max-Y_min) + Y_min))
    X0 = int(np.round(rand(1)*(X_max-X_min) + X_min))
    
    return (X0, Y0, X0+W_crop, Y0+H_crop)

def extract_random_patch(im, patch_size=(224, 224), border=(0, 0)):
    """
    Extract a random image patch of size `patch_size`,
    with the center of the patch inside `border`

    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
    :return:      np.ndarray
    """
    (X0, Y0, X1, Y1) = get_random_patch_dims(im, patch_size, border)    
    return im[Y0:Y1, X0:X1, ]

def cropout_patch(im, patch_size=(224, 224),
                  patch_position=(0.5, 0.5), fill_val=0):
    """
    Cropout (replace) a patch of size `patch_size` with `fill_val`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    * fill_val:       value to fill into the cropout
    :return:          np.ndarray
    """
    (X0, Y0, X1, Y1) = get_patch_dims(im, patch_size, patch_position)
    im_ = im.copy()
    im_[Y0:Y1, X0:X1, ] = fill_val
    return im_

def cropout_random_patch(im, patch_size=(224, 224), border=(0, 0), fill_val=0):
    """
    Cropout (replace) a random patch of size `patch_size` with `fill_val`,
    with the center of the patch inside `border`

    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
    * fill_val:   value to fill into the cropout
    :return:      np.ndarray
    """
    (X0, Y0, X1, Y1) = get_random_patch_dims(im, patch_size, border)
    im_ = im.copy()
    im_[Y0:Y1, X0:X1, ] = fill_val    
    return im_

# modified from stackoverflow
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(old_div(angle, (old_div(math.pi, 2))))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = old_div(d * math.sin(alpha), math.sin(delta))

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

# shuffle tiled images

def image_to_tiles(im, num_patches):
    """
    Cut an image `im` into equal sized patches.
    
    * im: input image
    * num_patches: (num_vertical, num_horizontal)
    """
    H, W = im.shape[:2]    
    H_count, W_count = num_patches
    patch_H, patch_W = int(H / H_count), int(W / W_count)
    tiles = [im[x:x+patch_H,y:y+patch_W] for x in range(0,H,patch_H) 
                                         for y in range(0,W,patch_W)]
    return tiles

def tiles_to_image(tiles, num_patches):
    """
    Reconstruct an image from tiles, resulting from `image_to_tiles`.
    
    * tiles: list of image patches
    * num_patches: (num_vertical, num_horizontal)
    """
    H_count, W_count = num_patches
    rows = [np.concatenate(row_list, axis=1) for row_list in chunks(tiles, W_count)]
    return np.concatenate(rows)

def imshuffle(im, num_patches):
    """
    Cut image into patches, shuffle and return the shuffled reconstructed image.
    Uses `image_to_tiles` and `tiles_to_image`.
    
    * im: input image
    * num_patches: (num_vertical, num_horizontal)
    """
    t = image_to_tiles(im, num_patches)
    np.random.shuffle(t)
    return tiles_to_image(t, num_patches)

def imshuffle_pair(im1, im2, num_patches, ratio=0.5, flip=False):
    """
    Scramble patches coming from two images into a single image.
    Similar to `imshuffle`, but the patches come from images `im1` and `im2`. 
    
    * im1, im2: input images, of equal size
    * num_patches: (num_vertical, num_horizontal) patches to divide each image in
    * ratio: fraction of patches to take from `im1`, the rest are taken from `im2`
    """
    t1 = image_to_tiles(im1, num_patches)
    t2 = image_to_tiles(im2, num_patches)
    np.random.shuffle(t1)
    np.random.shuffle(t2)
    counts = np.int32(np.round(len(t1)*ratio))
    t12 = t1[:counts] + t2[counts:]
    np.random.shuffle(t12)
    if flip:
        for i, _ in enumerate(t12):
            if rand(1) > 0.5: 
                t12[i] = np.fliplr(t12[i])
    return tiles_to_image(t12, num_patches)