"""STSnet model"""

from __future__ import division, print_function, absolute_import
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    AveragePooling3D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.layers import SpatialDropout2D, TimeDistributed
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Flatten, Permute, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


def STSNetModel(
    n_classes,
    n_channels=14,
    n_features=300,
    grid_rows=7,
    grid_cols=2,
    F1=8,
    D=2,
    F2=16,
    proj_k=9,
    dropout=0.25,
):
    """
    Input: (N, n_channels, n_features)   # no time axis, tabular features
    Map 14 channels -> 7x2 grid, features stay as the 'temporal-like' axis.
    """

    assert grid_rows * grid_cols >= n_channels

    inp = Input(shape=(n_channels, n_features))  # (C, F)
    # place channels on a 2D grid; pad extra cells if grid>channels
    x = Reshape((1, n_channels, n_features))(inp)  # (1, C, F)
    # pad to (grid_rows, grid_cols) along 'C' dimension:
    pad = grid_rows * grid_cols - n_channels
    if pad > 0:
        # Zero-pad channels dimension: (1, C+pad, F)
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, pad)))(x)

    x = Reshape((1, grid_rows * grid_cols, n_features))(x)
    x = Reshape((1, grid_rows, grid_cols * n_features))(
        x
    )  # fold cols*features for a simple start

    # Project along the (features) direction with 1xproj_k conv (channel-wise feature filter)
    x = Conv2D(F1, (1, proj_k), padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("elu")(x)

    # Spatial depthwise over (grid_rows, grid_cols) effect: use a depthwise with large kernel over rows
    # (simple variant to mix spatially without time). Kernel (grid_rows, 1) to mix rows; repeat for cols if desired.
    x = DepthwiseConv2D((grid_rows, 1), depth_multiplier=D, use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("elu")(x)

    # Feature mixing with separable conv along the (collapsed columns/features) axis
    x = SeparableConv2D(F2, (1, 7), padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("elu")(x)

    x = GlobalAveragePooling2D(data_format="channels_first")(x)
    x = Dropout(dropout)(x)
    out = Dense(n_classes, kernel_constraint=max_norm(0.25), activation="softmax")(x)

    return Model(inp, out)
