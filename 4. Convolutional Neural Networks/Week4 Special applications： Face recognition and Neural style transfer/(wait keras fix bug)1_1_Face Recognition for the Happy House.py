from fr_utils import *
from inception_blocks import *
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

import cv2
import os
import sys
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf


if __name__ == '__main__':
    #Todo channels_first 在2.2.4中没有修复 不想退到2.1.6 先暂停
    np.set_printoptions(threshold=sys.maxsize)
    K.set_image_data_format('channels_first')
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())


