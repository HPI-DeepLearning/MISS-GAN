"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import scipy.misc
import pdb
import numpy as np
from time import gmtime, strftime

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, image_size, input_c_dim, output_c_dim, is_train=False):
    input_img = imread(image_path)
    images = np.split(input_img, input_c_dim + output_c_dim, axis=1)

    half_offset = 8
    offset = half_offset * 2
    hypersize = image_size + offset
    fullsize = 512 + offset

    h1 = int(np.ceil(np.random.uniform(1e-2, offset)))
    w1 = int(np.ceil(np.random.uniform(1e-2, offset)))

    conv = []
    for image in images:
        #print(image.shape)
        top = int((fullsize - image.shape[1]) / 2)
        bottom = fullsize - image.shape[1] - top
        image = np.append(np.zeros((image.shape[0], top)), image, axis=1)
        image = np.append(image, np.zeros((image.shape[0], bottom)), axis=1)

        left = int((fullsize - image.shape[0]) / 2)
        right = fullsize - image.shape[0] - left
        image = np.append(np.zeros((left, image.shape[1])), image, axis=0)
        image = np.append(image, np.zeros((right, image.shape[1])), axis=0)

        tmp = scipy.misc.imresize(image, [hypersize, hypersize], interp='nearest')
        if is_train:
            image = tmp[h1:h1+image_size, w1:w1+image_size]
        else:
            image = tmp[half_offset:half_offset+image_size, half_offset:half_offset+image_size]
        image = image/127.5 - 1.
		
        conv.append(image)

    return np.stack(conv, axis=2)

# -----------------------------

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.
