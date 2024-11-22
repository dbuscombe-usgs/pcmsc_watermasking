# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-24, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json, os
# from doodleverse_utils.imports import *
# from scipy.ndimage import maximum_filter
from skimage.transform import resize
from skimage.filters import threshold_otsu
# from joblib import Parallel, delayed
from skimage import io
from skimage.io import imread, imsave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#####################################################
#### definitions
# =========================================================
import tensorflow as tf  # numerical operations on gpu
import tensorflow.keras.backend as K
from transformers import TFSegformerForSemanticSegmentation


SEED = 42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE  # used in tf.data.Dataset API

tf.random.set_seed(SEED)



##========================================================
def fromhex(n):
    """hexadecimal to integer"""
    return int(n, base=16)


##========================================================
def label_to_colors(
    img,
    mask,
    alpha,  # =128,
    colormap,  # =class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,  # =0,
    do_alpha,  # =True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask == 1] = (0, 0, 0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg


###############################################################
### MODEL SUBFUNCTIONS
###############################################################

# -----------------------------------
def upsamp_concat_block(x, xskip):
    """
    upsamp_concat_block(x, xskip)
    This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
    INPUTS:
        * `xskip`: input keras layer (skip connection)
        * `x`: input keras layer
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    u = tf.keras.layers.UpSampling2D((2, 2))(x)

    return tf.keras.layers.Concatenate()([u, xskip])


# -----------------------------------
def conv_block(
    x,
    filters,
    kernel_size=(7, 7),
    padding="same",
    strides=1,
    dropout=0.1,
    dropout_type="standard",
):
    """
    conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
    This function applies batch normalization to an input layer, then convolves with a 2D convol layer
    The two actions combined is called a convolutional block

    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`:input keras layer to be convolved by the block
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the batch normalized convolution
    """

    if dropout_type == "spatial":
        DO = tf.keras.layers.SpatialDropout2D
    elif dropout_type == "standard":
        DO = tf.keras.layers.Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    if dropout > 0.0:
        x = DO(dropout)(x)

    conv = batchnorm_act(x)
    return tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides
    )(conv)


# -----------------------------------
def bottleneck_block(x, filters, kernel_size=(2, 2), padding="same", strides=1):
    """
    bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

    This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between convolutional and bottleneck layers
    """
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides
    )(x)
    conv = conv_block(
        conv,
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dropout=0.0,
        dropout_type="standard",
    )

    bottleneck = tf.keras.layers.Conv2D(
        filters, kernel_size=(1, 1), padding=padding, strides=strides
    )(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([conv, bottleneck])


def res_block(
    x,
    filters,
    kernel_size=(7, 7),
    padding="same",
    strides=1,
    dropout=0.1,
    dropout_type="standard",
):
    """
    res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
    This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
    INPUTS:
        * `filters`: number of filters in the convolutional block
        * `x`: input keras layer
    OPTIONAL INPUTS:
        * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras layer, output of the addition between residual convolutional and bottleneck layers
    """
    res = conv_block(
        x,
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        dropout=dropout,
        dropout_type=dropout_type,
    )
    res = conv_block(
        res,
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=1,
        dropout=dropout,
        dropout_type=dropout_type,
    )

    bottleneck = tf.keras.layers.Conv2D(
        filters, kernel_size=(1, 1), padding=padding, strides=strides
    )(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])


# -----------------------------------
def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.1,
    dropout_type="standard",
    filters=16,
    kernel_size=(2, 2),
    activation="relu",
    strides=(1, 1),
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = tf.keras.layers.SpatialDropout2D
    elif dropout_type == "standard":
        DO = tf.keras.layers.Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        strides=strides,
        use_bias=not use_batch_norm,
    )(inputs)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)

    c = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        strides=strides,
        use_bias=not use_batch_norm,
    )(c)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)
    return c


# -----------------------------------
def res_conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.1,
    dropout_type="standard",
    filters=16,
    kernel_size=(2, 2),
    activation="relu",
    strides=(1, 1),
    kernel_initializer="he_normal",
    padding="same",
):

    res = conv2d_block(
        inputs=inputs,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        strides=strides,
        kernel_initializer="he_normal",
        padding="same",
    )

    res = conv2d_block(
        inputs=res,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        strides=(1, 1),
        kernel_initializer="he_normal",
        padding="same",
    )

    bottleneck = tf.keras.layers.Conv2D(
        filters, kernel_size=(1, 1), padding=padding, strides=strides
    )(
        inputs
    )  ##kernel_size
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])


# -----------------------------------
def batchnorm_act(x):
    """
    batchnorm_act(x)
    This function applies batch normalization to a keras model layer, `x`, then a relu activation function
    INPUTS:
        * `z` : keras model layer (should be the output of a convolution or an input layer)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * batch normalized and relu-activated `x`
    """
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


###############################################################
### LOSSES AND METRICS
###############################################################

# -----------------------------------

#define the basic IOU formula. 
def basic_iou(y_true, y_pred):
    smooth = 10e-6
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union =  tf.reduce_sum(y_true_f + y_pred_f) - intersection
    return (intersection+smooth)/(union+ smooth)

#define the IoU metric for nclasses
def iou_multi(nclasses):
    """
    mean_iou(y_true, y_pred)
    This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * IoU score [tensor]
    """
    def mean_iou(y_true, y_pred):
        iousum = 0
        y_pred = tf.one_hot(tf.argmax(y_pred, -1), nclasses)
        for index in range(nclasses):
            iousum += basic_iou(y_true[:,:,:,index], y_pred[:,:,:,index])
        return iousum/nclasses

    return mean_iou

# -----------------------------------
#define basic Dice formula
# @tf.autograph.experimental.do_not_convert
def basic_dice_coef(y_true, y_pred):
    """
    dice_coef(y_true, y_pred)

    This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice score [tensor]
    """
    smooth = 10e-6
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice

#define Dice formula for multiple classes
# @tf.autograph.experimental.do_not_convert
def dice_multi(nclasses):

    def dice_coef(y_true, y_pred):
        dice = 0
        #can't have an argmax in a loss
        for index in range(nclasses):
            dice += basic_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return dice/nclasses

    return dice_coef

# ---------------------------------------------------
#define Dice loss for multiple classes
# @tf.autograph.experimental.do_not_convert
def dice_coef_loss(nclasses):
    """
    dice_coef_loss(y_true, y_pred)

    This function computes the mean Dice loss (1 - Dice coefficient) between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice loss [tensor]
    """
    def MC_dice_coef_loss(y_true, y_pred):
        dice = 0
        #can't have an argmax in a loss
        for index in range(nclasses):
            dice += basic_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return 1 - (dice/nclasses)

    return MC_dice_coef_loss

#define weighted Dice loss for multiple classes
# @tf.autograph.experimental.do_not_convert
def weighted_dice_coef_loss(nclasses, weights):
    """
    weighted_MC_dice_coef_loss(y_true, y_pred)

    This function computes the mean Dice loss (1 - Dice coefficient) between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice loss [tensor]
    """

    def weighted_MC_dice_coef_loss(y_true, y_pred):
        dice = 0
        #can't have an argmax in a loss
        for index in range(nclasses):
            dice += basic_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])*weights[index]
        meandice = (dice/nclasses)
        return 1 - meandice

    return weighted_MC_dice_coef_loss

def mean_iou_np(y_true, y_pred, nclasses):
    iousum = 0
    y_pred = tf.one_hot(tf.argmax(y_pred, -1), nclasses)
    for index in range(nclasses):
        iousum += basic_iou(y_true[:,:,:,index], y_pred[:,:,:,index])
    return (iousum/nclasses).numpy()


def mean_dice_np(y_true, y_pred, nclasses):
    dice = 0
    #can't have an argmax in a loss
    for index in range(nclasses):
        dice += basic_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return (dice/nclasses).numpy()



# -----------------------------------
def custom_resunet(
    sz,
    f,
    nclasses=2,
    kernel_size=(7, 7),
    strides=2,
    dropout=0.1,
    dropout_change_per_layer=0.0,
    dropout_type="standard",
    use_dropout_on_upsampling=False,
):
    """
    res_unet(sz, f, nclasses=1)
    This function creates a custom residual U-Net model for image segmentation
    INPUTS:
        * `sz`: [tuple] size of input image
        * `f`: [int] number of filters in the convolutional block
        * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
        * nclasses [int]: number of classes
        dropout (float , 0. and 1.): dropout after the first convolutional block. 0. = no dropout

        dropout_change_per_layer (float , 0. and 1.): Factor to add to the Dropout after each convolutional block

        dropout_type (one of "spatial" or "standard"): Spatial is recommended  by  https://arxiv.org/pdf/1411.4280.pdf

        use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network

        filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    OPTIONAL INPUTS:
        * `kernel_size`=(7, 7): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
        * `padding`="same":  see tf.keras.layers.Conv2D
        * `strides`=1: see tf.keras.layers.Conv2D
    GLOBAL INPUTS: None
    OUTPUTS:
        * keras model
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f)
    f = int(f * 2)
    e2 = res_block(
        e1,
        f,
        strides=strides,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )
    f = int(f * 2)
    dropout += dropout_change_per_layer
    e3 = res_block(
        e2,
        f,
        strides=strides,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )
    f = int(f * 2)
    dropout += dropout_change_per_layer
    e4 = res_block(
        e3,
        f,
        strides=strides,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )
    f = int(f * 2)
    dropout += dropout_change_per_layer
    _ = res_block(
        e4,
        f,
        strides=strides,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )

    ## bottleneck
    b0 = conv_block(
        _,
        f,
        strides=1,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )
    _ = conv_block(
        b0,
        f,
        strides=1,
        kernel_size=kernel_size,
        dropout=dropout,
        dropout_type=dropout_type,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(
        _, f, kernel_size=kernel_size, dropout=dropout, dropout_type=dropout_type
    )
    f = int(f / 2)
    dropout -= dropout_change_per_layer

    _ = upsamp_concat_block(_, e3)
    _ = res_block(
        _, f, kernel_size=kernel_size, dropout=dropout, dropout_type=dropout_type
    )
    f = int(f / 2)
    dropout -= dropout_change_per_layer

    _ = upsamp_concat_block(_, e2)
    _ = res_block(
        _, f, kernel_size=kernel_size, dropout=dropout, dropout_type=dropout_type
    )
    f = int(f / 2)
    dropout -= dropout_change_per_layer

    _ = upsamp_concat_block(_, e1)
    _ = res_block(
        _, f, kernel_size=kernel_size, dropout=dropout, dropout_type=dropout_type
    )

    outputs = tf.keras.layers.Conv2D(
        nclasses, (1, 1), padding="same", activation="softmax"#, dtype='float32'
    )(_)

    # model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model


###############################################################
def segformer(
    id2label,
    num_classes=2,
):
    """
    https://keras.io/examples/vision/segformer/
    https://huggingface.co/nvidia/mit-b0
    """

    label2id = {label: id for id, label in id2label.items()}
    model_checkpoint = "nvidia/mit-b0"

    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model

##====================================
def standardize(img):
    # standardization using adjusted standard deviation

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))

    return img

# =========================================================
def return_img_array(f, TARGET_SIZE, MODEL):
    if 'jpg' in f:
        segfile = f.replace('.jpg', '_seg.tif')
    elif 'png' in f:
        segfile = f.replace('.png', '_seg.tif')

    segfile = os.path.normpath(segfile)

    image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
    image_stan = standardize(image.numpy()).squeeze()

    if MODEL=='segformer':
        if np.ndim(image)==2:
            image_stan = np.dstack((image_stan, image_stan, image_stan))
        image_stan = tf.transpose(image_stan, (2, 0, 1))


    return image_stan, w, h, segfile, bigimage

##==================================
def file2array(f):
    # bigimage = imread(f) ###slow!
    return  io.imread(f)

# #-----------------------------------
def seg_file2tensor_3band(f,TARGET_SIZE):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    # bigimage = imread(f) ###slow!
    bigimage = io.imread(f)

    M = bigimage.shape[0]//2
    N = bigimage.shape[1]//2

    smallimage = resize(bigimage,(TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True)

    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage 

# =========================================================

def get_thres(out_stack):

    # thres_land = 0.5 #threshold_otsu(out_stack[:,:,0])
    # thres_conf = np.median(out_stack[:,:,1].flatten()) #threshold_otsu(out_stack[:,:,1])
    # thres_var = np.median(out_stack[:,:,2].flatten()) # threshold_otsu(out_stack[:,:,2])
    thres_land = threshold_otsu(out_stack[:,:,0])
    thres_conf = threshold_otsu(out_stack[:,:,1])
    thres_var = threshold_otsu(out_stack[:,:,2])

    return thres_land, thres_conf, thres_var


# =========================================================
def get_prob_stacks_sand(M, image_stan, w, h, MODEL, NCLASSES, TARGET_SIZE):

    # w = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(run_model(image_stan, model))() for model in M)
    # E0, E1 = zip(*w)

    E0 = []; E1 = [];

    for model in M:

        sand, nosand = run_model_sand(image_stan, model, w, h, MODEL, NCLASSES, TARGET_SIZE)

        E0.append(sand)
        E1.append(nosand)
        # del e0, e1
    return E0, E1


# =========================================================
def get_prob_stacks(M, image_stan, w, h, MODEL, NCLASSES, TARGET_SIZE):

    # w = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(run_model(image_stan, model))() for model in M)
    # E0, E1 = zip(*w)

    E0 = []; E1 = [];

    for model in M:

        e0, e1 = run_model(image_stan, model, w, h, MODEL, NCLASSES, TARGET_SIZE)

        E0.append(e0)
        E1.append(e1)
        # del e0, e1
    return E0, E1


# =========================================================
def run_model(image_stan, model, w, h, MODEL, NCLASSES, TARGET_SIZE):

    try:
        if MODEL=='segformer':
            est_label = model(tf.expand_dims(image_stan, 0)).logits
        else:
            est_label = tf.squeeze(model.predict(tf.expand_dims(image_stan, 0), batch_size=1))

    except:
        if MODEL=='segformer':
            est_label = model.predict(tf.expand_dims(image_stan[:,:,0], 0), batch_size=1).logits
        else:
            est_label = tf.squeeze(model.predict(tf.expand_dims(image_stan[:,:,0], 0), batch_size=1))

    if MODEL=='segformer':
        est_label = tf.nn.softmax(est_label, axis=1)

    est_label = tf.squeeze(est_label).numpy()

    if MODEL=='segformer':
        est_label = np.transpose(est_label, (1,2,0))
        est_label = resize(est_label, (TARGET_SIZE[0],TARGET_SIZE[1],NCLASSES), preserve_range=True, clip=True).squeeze()
        # est_label = rescale_array(est_label,0,1)

    # print(np.max(est_label))
    # print(np.min(est_label))

    e0 = resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True)
    e1 = resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True)
    # e1 = maximum_filter(e1,(3,3))

    return e0, e1


##========================================================
def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


# =========================================================
def run_model_sand(image_stan, model, w, h, MODEL, NCLASSES, TARGET_SIZE):

    try:
        if MODEL=='segformer':
            est_label = model(tf.expand_dims(image_stan, 0)).logits
        else:
            est_label = tf.squeeze(model.predict(tf.expand_dims(image_stan, 0), batch_size=1))

    except:
        if MODEL=='segformer':
            est_label = model.predict(tf.expand_dims(image_stan[:,:,0], 0), batch_size=1).logits
        else:
            est_label = tf.squeeze(model.predict(tf.expand_dims(image_stan[:,:,0], 0), batch_size=1))


    if MODEL=='segformer':
        est_label = tf.nn.softmax(est_label, axis=1)

    est_label = tf.squeeze(est_label).numpy()

    if MODEL=='segformer':
        est_label = np.transpose(est_label, (1,2,0))
        est_label = resize(est_label, (TARGET_SIZE[0],TARGET_SIZE[1],NCLASSES), preserve_range=True, clip=True).squeeze()
        # est_label = rescale_array(est_label,0,1)

    sand = resize(est_label[:,:,2],(w,h), preserve_range=True, clip=True)

    tmp = np.delete(est_label,axis=-1,obj=1)
    nosand = resize(np.max(tmp,-1),(w,h), preserve_range=True, clip=True)

    return sand, nosand



# =========================================================

def average_probstack(E0,E1,WEIGHTING):

    tmp = np.dstack(E0)
    e0 = np.average(tmp, axis=-1, weights=np.array(WEIGHTING))

    var0 = np.std(tmp, axis=-1)

    tmp = np.dstack(E1)
    e1 = np.average(tmp, axis=-1, weights=np.array(WEIGHTING))

    var1 = np.std(tmp, axis=-1)

    est_label = np.maximum(e1, 1-e0)

    conf=1-np.minimum(e0,e1)

    return np.dstack((est_label,conf,var0+var1))


# =========================================================
def do_seg(f, Mc, MODEL, WEIGHTING, TARGET_SIZE, meta, out_folder, logic=6):
    """
    Carries out image segmentation of image file f (str), using model M (a list of keras objects),
    weighting specified by WEIGHTING (list of floats), and results stored to 'out_folder' (str)

    f may be a 'jpg' or 'png' file extension

    M is a list of models that have already been compiled and are ready for inference. 
    The program is set up to use an ensemble of models. A weighted average is applied to model outputs. 
    So if M is a list of  3 models, model 1, 2, and 3 softmax scores will be added, 
    then a weighted average will be carried out according to WEIGHTING,
    a list of floats for each model

    a 'maximum filter' is applied to softmax scores before averaging, using a 3x3 pixel kernel 

    meta is a dictionary of metadata associated with the file f

    'out_folder' is a full directory path (str) to store outputs. 
    Subdirectories will be made by the program, as necessary
    """

    NCLASSES=2
    meta['image_filename'] = f.split(os.sep)[-1]

    image_stan, w, h, segfile, bigimage = return_img_array(f, TARGET_SIZE, MODEL)

    E0, E1 = get_prob_stacks(Mc, image_stan, w, h, MODEL, NCLASSES, TARGET_SIZE)

    K.clear_session()

    out_stack = average_probstack(E0,E1,WEIGHTING)

    thres_land, thres_conf, thres_var = get_thres(out_stack)
    # print(thres_land)
    # print(thres_conf)
    # print(thres_var)

    meta['otsu_land'] = thres_land
    meta['otsu_confidence'] = thres_conf
    meta['otsu_variance'] = thres_var

    if logic==6:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8') + (out_stack[:,:,1]<thres_conf).astype('uint8') + (out_stack[:,:,2]>thres_var).astype('uint8')
        mask[mask>1]=1
    elif logic==5:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8') + (out_stack[:,:,1]<thres_conf).astype('uint8') 
        mask[mask>1]=1
    elif logic==4:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8')
        mask[mask>1]=1

    print_figs(segfile, bigimage, out_stack, mask, out_folder, meta)



# =========================================================
def do_seg_sand(f, Mc, MODEL, WEIGHTING, TARGET_SIZE, meta, out_folder, logic=6):
    """
    Carries out image segmentation of image file f (str), using model M (a list of keras objects),
    weighting specified by WEIGHTING (list of floats), and results stored to 'out_folder' (str)

    f may be a 'jpg' or 'png' file extension

    M is a list of models that have already been compiled and are ready for inference. 
    The program is set up to use an ensemble of models. A weighted average is applied to model outputs. 
    So if M is a list of  3 models, model 1, 2, and 3 softmax scores will be added, 
    then a weighted average will be carried out according to WEIGHTING,
    a list of floats for each model

    a 'maximum filter' is applied to softmax scores before averaging, using a 3x3 pixel kernel 

    meta is a dictionary of metadata associated with the file f

    'out_folder' is a full directory path (str) to store outputs. 
    Subdirectories will be made by the program, as necessary
    """

    NCLASSES=8 ## use CT8 modfel
    # NCLASSES=5 ## use CT5 modfel

    meta['image_filename'] = f.split(os.sep)[-1]

    image_stan, w, h, segfile, bigimage = return_img_array(f, TARGET_SIZE, MODEL)

    E0, E1 = get_prob_stacks_sand(Mc, image_stan, w, h, MODEL, NCLASSES, TARGET_SIZE)

    K.clear_session()

    out_stack = average_probstack(E0,E1,WEIGHTING)

    thres_land, thres_conf, thres_var = get_thres(out_stack)
    
    meta['otsu_land'] = thres_land
    meta['otsu_confidence'] = thres_conf
    meta['otsu_variance'] = thres_var

    if logic==6:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8') + (out_stack[:,:,1]<thres_conf).astype('uint8') + (out_stack[:,:,2]>thres_var).astype('uint8')
        mask[mask>1]=1
    elif logic==5:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8') + (out_stack[:,:,1]<thres_conf).astype('uint8') 
        mask[mask>1]=1
    elif logic==4:
        mask =  (out_stack[:,:,0]>thres_land).astype('uint8')
        mask[mask>1]=1

    print_figs_sand(segfile, bigimage, out_stack, mask, out_folder, meta)



# =========================================================
def print_figs(segfile, bigimage, out_stack, mask6, out_folder, meta):

    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'water_prob_stack'))

    imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'water_masks'))

    imsave(outfile.replace('.tif','.jpg'),255*mask6.astype('uint8'),quality=100)

    #====================
    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(mask6, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(mask6, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)


    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'water_mask_overlays'))

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=100, bbox_inches='tight')
    plt.close('all')

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'water_meta'))
		
    np.savez_compressed(outfile.replace('.tif','.npz'), **meta)


# =========================================================
def print_figs_sand(segfile, bigimage, out_stack, mask6, out_folder, meta):

    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'sand_prob_stack'))

    imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'sand_masks'))

    imsave(outfile.replace('.tif','.jpg'),255*mask6.astype('uint8'),quality=100)

    #====================
    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(mask6, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(mask6, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)


    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'sand_mask_overlays'))

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=100, bbox_inches='tight')
    plt.close('all')

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'sand_meta'))
		
    np.savez_compressed(outfile.replace('.tif','.npz'), **meta)

