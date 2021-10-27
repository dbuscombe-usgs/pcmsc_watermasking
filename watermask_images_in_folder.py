# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
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

import os, time

USE_GPU = True #False

if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk
from tkinter import filedialog
from tkinter import *
import json
from skimage.io import imsave, imread
from numpy.lib.stride_tricks import as_strided as ast
# from skimage.exposure import adjust_log, adjust_gamma

from joblib import Parallel, delayed
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import maximum_filter
from skimage.transform import resize
from tqdm import tqdm
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

import tensorflow as tf #numerical operations on gpu
import tensorflow.keras.backend as K


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels

# #-----------------------------------
def seg_file2tensor_3band(f):#, resize):
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

    bigimage = imread(f)#Image.open(f)

    M = bigimage.shape[0]//2
    N = bigimage.shape[1]//2
    # tiles = [bigimage[x:x+M,y:y+N] for x in range(0,bigimage.shape[0],M) for y in range(0,bigimage.shape[1],N)]
    #
    # tiles = [resize(t,(TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True) for t in tiles]

    smallimage = resize(bigimage,(TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True)

    # smallimage = adjust_log(smallimage,gain=1.2)
    #
    # smallimage = rescale(standardize(smallimage),0,255)
    smallimage = tf.cast(smallimage, tf.uint8)

    # tiles = [tf.cast(t, tf.uint8) for t in tiles]
    # tiles = [standardize(t/255.) for t in tiles]

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage ##tiles

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    mask,
    alpha,#=128,
    colormap,#=class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,#=0,
    do_alpha,#=True
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

    cimg[mask==1] = (0,0,0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg


# =========================================================
def do_seg(f, models, W, meta):

    meta['image_filename'] = f.split(os.sep)[-1]

    if 'jpg' in f:
        segfile = f.replace('.jpg', '_seg.tif')
    elif 'png' in f:
        segfile = f.replace('.png', '_seg.tif')

    segfile = os.path.normpath(segfile)

    # if os.path.exists(segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'certainty'))):
    #     print('%s exists ... skipping' % (segfile))
    #     pass
    # else:
    #     print('%s does not exist ... creating' % (segfile))

    start = time.time()

    image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
    image = standardize(image.numpy()).squeeze()


    E0 = []; E1 = []; #W = []

    for counter,model in enumerate(models):

        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        # E0.append(est_label[:,:,0])
        # E1.append(est_label[:,:,1])
        print('Model {} applied'.format(counter))
        E0.append(resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True))
        E1.append(resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True))
        del est_label

        # est_label = np.argmax(est_label, -1)
        # E.append(est_label)
        # W.append(1)

        # for k in np.linspace(100,int(TARGET_SIZE[0]),3):
        #     est_label = model.predict(tf.expand_dims(np.roll(image, int(k)), 0) , batch_size=1).squeeze()
        #     E0.append(est_label[:,:,0])
        #     E1.append(est_label[:,:,1])
        #     W.append(2*(1/np.sqrt(k)))

    print('Models applied')
    K.clear_session()
    del image


    # outfile = TemporaryFile()
    # fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
    # fp[:] = E0[:]
    # fp.flush()
    # del E0
    # del fp

    # E0 = [resize(e,(w,h), preserve_range=True, clip=True) for e in E0]
    e0 = np.average(np.dstack(E0), axis=-1, weights=np.array(W))

    var0 = np.std(np.dstack(E0), axis=-1)

    del E0

    # E1 = [resize(e,(w,h), preserve_range=True, clip=True) for e in E1]
    e1 = np.average(np.dstack(E1), axis=-1, weights=np.array(W))

    var1 = np.std(np.dstack(E1), axis=-1)

    del E1


    # thres0 = threshold_otsu(e0)
    # # print("Probability of water threshold: %f" % (thres0))
    #
    # thres1 = threshold_otsu(e1)
    # # print("Probability of land threshold: %f" % (thres1))

    # est_label = np.zeros_like(e1)#+0.5
    # est_label[e1>np.maximum(thres0,thres1)] = 1
    # est_label[e1<np.minimum(thres0,thres1)] = 0
    # est_label = np.argmax(np.dstack((e0,e1)),-1)

    est_label = (e1+(1-e0))/2

    conf=1-np.minimum(e0,e1)

    del e0, e1

    est_label = maximum_filter(est_label,(7,7))

    print('Probability of land computed')

    # plt.imshow(bigimage); plt.imshow(est_label, alpha=0.5); plt.show()

    out_stack = np.dstack((est_label,conf,var0+var1))
    del est_label, conf, var0, var1

    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'prob_stack'))

    try:
    	os.mkdir(os.path.normpath(sample_direc+os.sep+'prob_stack'))
    except:
    	pass

    imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)


    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'probstack_overlays'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'probstack_overlays'))
    except:
        pass

    plt.imshow(bigimage); plt.imshow(out_stack, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=200, bbox_inches='tight')
    plt.close('all')

    print('Probability stack computed')

    ####===============================================================
    temp = -0.05
    thres_land = threshold_otsu(out_stack[:,:,0])+temp
    thres_conf = threshold_otsu(out_stack[:,:,1])
    thres_var = threshold_otsu(out_stack[:,:,2])
    print("Land threshold: %f" % (thres_land))
    print("Confidence threshold: %f" % (thres_conf))
    print("Variance threshold: %f" % (thres_var))

    #mask1 (conservative)
    mask1 = (out_stack[:,:,0]>thres_land) & (out_stack[:,:,1]>thres_conf) & (out_stack[:,:,2]<thres_var)
    mask2 = (out_stack[:,:,0]>thres_land) & (out_stack[:,:,2]<thres_var)
    mask3 = (out_stack[:,:,0]>thres_land) & (out_stack[:,:,1]>thres_conf)
    mask4 = (out_stack[:,:,0]>thres_land)
    # del var0, var1, est_label, conf

    #ultra-conservative
    mask0 = ((mask1+mask2+mask3+mask4)==4).astype('uint8')

    land = (out_stack[:,:,0]>thres_land)
    # land = (conf>thres_conf)

    del out_stack

    print('Land masks computed')

    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks0'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks0'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask0.astype('uint8'),quality=100)
    del mask0

    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks1'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks1'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask1.astype('uint8'),quality=100)
    del mask1


    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks2'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks2'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask2.astype('uint8'),quality=100)
    del mask2


    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks3'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks3'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask3.astype('uint8'),quality=100)
    del mask3


    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks4'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks4'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask4.astype('uint8'),quality=100)
    del mask4


    nx,ny=np.shape(land)
    island_thres = 100*np.maximum(nx,ny)

    land = remove_small_holes(land.astype('bool'), island_thres)
    land = remove_small_objects(land, island_thres).astype('uint8')

    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*land.astype('uint8'),quality=100)


    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(land, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(land, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

    del land

    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'overlays'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'overlays'))
    except:
        pass

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=200, bbox_inches='tight')
    plt.close('all')


    # plt.imshow(bigimage); plt.imshow(land, alpha=0.5); plt.show()

    #land = remove_small_holes(land.astype('uint8'), 5*w)
    #land = remove_small_objects(land.astype('uint8'), 5*w)

    del bigimage, color_label

    print('Outputs made')

    elapsed = (time.time() - start)/60
    print("Image masking took "+ str(elapsed) + " minutes")

#====================================================

root = Tk()
root.filename =  filedialog.askdirectory(initialdir = "/samples",title = "Select directory of images to segment")
sample_direc = root.filename
print(sample_direc)
root.withdraw()

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/config/",title = "Select file",filetypes = (("config DEPLOYMENT file","*.json"),("all files","*.*")))
configfile = root.filename
print(configfile)
root.withdraw()


# configfile = weights.replace('.h5','.json').replace('weights', 'config')


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')


from imports import *

#=======================================================
# model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), BATCH_SIZE, NCLASSES)

print('.....................................')
print('Creating and compiling models...')

meta = dict()

weights = []
if 'WEIGHTS1' in locals():
    weights.append(WEIGHTS1)
    meta['WEIGHTS1'] = WEIGHTS1
if 'WEIGHTS2' in locals():
    weights.append(WEIGHTS2)
    meta['WEIGHTS2'] = WEIGHTS2
if 'WEIGHTS3' in locals():
    weights.append(WEIGHTS3)
    meta['WEIGHTS3'] = WEIGHTS3
if 'WEIGHTS4' in locals():
    weights.append(WEIGHTS4)
    meta['WEIGHTS4'] = WEIGHTS4
if 'WEIGHTS5' in locals():
    weights.append(WEIGHTS5)
    meta['WEIGHTS5'] = WEIGHTS5
if 'WEIGHTS6' in locals():
    weights.append(WEIGHTS6)
    meta['WEIGHTS6'] = WEIGHTS6

meta['TARGET_SIZE'] = TARGET_SIZE

models = []
for w in weights:

    if 'resunet' in w:
        # num_filters = 8 # initial filters
        # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))
        NCLASSES= 1
        FILTERS=10
        N_DATA_BANDS= 3
        UPSAMPLE_MODE="simple"
        DROPOUT=0.1
        DROPOUT_CHANGE_PER_LAYER=0.0
        DROPOUT_TYPE="standard"
        USE_DROPOUT_ON_UPSAMPLING=False

        model = custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    upsample_mode=UPSAMPLE_MODE,#"deconv",
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #346,564
    elif 'unet' in w:

        NCLASSES= 1
        FILTERS=13
        N_DATA_BANDS= 3
        UPSAMPLE_MODE="simple"
        DROPOUT=0.1
        DROPOUT_CHANGE_PER_LAYER=0.0
        DROPOUT_TYPE="standard"
        USE_DROPOUT_ON_UPSAMPLING=False

        model = custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    upsample_mode=UPSAMPLE_MODE,#"deconv",
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #242,812

    elif 'satunet' in w:
        # model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

        NCLASSES= 1
        FILTERS=17
        N_DATA_BANDS= 3
        UPSAMPLE_MODE="simple"
        DROPOUT=0.1
        DROPOUT_CHANGE_PER_LAYER=0.0
        DROPOUT_TYPE="standard"
        USE_DROPOUT_ON_UPSAMPLING=False

        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    upsample_mode=UPSAMPLE_MODE,#"deconv",
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))

    # else:
    #     print("Model must be one of 'unet', 'resunet', or 'satunet'")
    #     sys.exit(2)

    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
    model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

    model.load_weights(w)

    models.append(model)


# W = [1,1,1,1,.5,.5]

W = [1 for m in models]

### predict
print('.....................................')
print('Using model for prediction on images ...')

sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
if len(sample_filenames)==0:
    sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):
    do_seg(f, models, W, meta)
    print('%i out of %i done'%(counter,len(sample_filenames)))


# w = Parallel(n_jobs=2, verbose=0, max_nbytes=None)(delayed(do_seg)(f) for f in tqdm(sample_filenames))
