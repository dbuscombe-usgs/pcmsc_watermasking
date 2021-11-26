# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-21, Marda Science LLC
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

USE_GPU =False

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
from glob import glob


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels



##========================================================
def crf_refine(label,
    img,
    crf_theta_slider_value,
    crf_mu_slider_value,
    crf_downsample_factor,
    gt_prob):
    """
    "crf_refine(label, img)"
    This function refines a label image based on an input label image and the associated image
    Uses a conditional random field algorithm using spatial and image features
    INPUTS:
        * label [ndarray]: label image 2D matrix of integers
        * image [ndarray]: image 3D matrix of integers
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: label [ndarray]: label image 2D matrix of integers
    """

    Horig = label.shape[0]
    Worig = label.shape[1]

    l_unique = np.unique(label.flatten())#.tolist()
    scale = 1+(5 * (np.array(img.shape).max() / 3000))

    # decimate by factor by taking only every other row and column
    img = img[::crf_downsample_factor,::crf_downsample_factor, :]
    # do the same for the label image
    label = label[::crf_downsample_factor,::crf_downsample_factor]
    # yes, I know this aliases, but considering the task, it is ok; the objective is to
    # make fast inference and resize the output

    Hnew = label.shape[0]
    Wnew = label.shape[1]

    orig_mn = np.min(np.array(label).flatten())
    orig_mx = np.max(np.array(label).flatten())

    if l_unique[0]==0:
        n = (orig_mx-orig_mn)#+1
    else:

        n = (orig_mx-orig_mn)+1
        label = (label - orig_mn)+1
        mn = np.min(np.array(label).flatten())
        mx = np.max(np.array(label).flatten())

        n = (mx-mn)+1

    H = label.shape[0]
    W = label.shape[1]
    U = unary_from_labels(label.astype('int'), n, gt_prob=gt_prob)
    d = dcrf.DenseCRF2D(H, W, n)
    d.setUnaryEnergy(U)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(3, 3),
                 compat=3,
                 kernel=dcrf.DIAG_KERNEL,
                 normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(
                          sdims=(crf_theta_slider_value, crf_theta_slider_value),
                          schan=(scale,scale,scale),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=crf_mu_slider_value, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC) #260

    Q = d.inference(10)
    result = np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8) +1

    uniq = np.unique(result.flatten())

    result = resize(result, (Horig, Worig), order=0, anti_aliasing=False) #True)

    result = rescale(result, orig_mn, orig_mx).astype(np.uint8)

    return result, n

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
def do_seg(f, M, WEIGHTING, meta):

    meta['image_filename'] = f.split(os.sep)[-1]

    if 'jpg' in f:
        segfile = f.replace('.jpg', '_seg.tif')
    elif 'png' in f:
        segfile = f.replace('.png', '_seg.tif')

    segfile = os.path.normpath(segfile)

    start = time.time()

    image, w, h, bigimage = seg_file2tensor_3band(f)#, resize=True)
    image_stan = standardize(image.numpy()).squeeze()

    do_crf_refine = True

    E0 = []; E1 = [];

    for counter,model in enumerate(M):

        est_label = model.predict(tf.expand_dims(image_stan, 0) , batch_size=1).squeeze()

        print('Model {} applied'.format(counter))
        e0 = resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True)
        e1 = resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True)
        e1 = maximum_filter(e1,(3,3))

        E0.append(e0)
        E1.append(e1)
        del est_label, e0, e1

    #print('Models applied')
    K.clear_session()
    del image_stan

    # outfile = TemporaryFile()
    # fp = np.memmap(outfile, dtype='uint8', mode='w+', shape=out_mask.shape)
    # fp[:] = E0[:]
    # fp.flush()
    # del E0
    # del fp

    # E0 = [resize(e,(w,h), preserve_range=True, clip=True) for e in E0]
    e0 = np.average(np.dstack(E0), axis=-1, weights=np.array(WEIGHTING))

    var0 = np.std(np.dstack(E0), axis=-1)

    del E0

    # E1 = [resize(e,(w,h), preserve_range=True, clip=True) for e in E1]
    e1 = np.average(np.dstack(E1), axis=-1, weights=np.array(WEIGHTING))

    var1 = np.std(np.dstack(E1), axis=-1)

    del E1

    #est_label = (e1+(1-e0))/2
    est_label = np.maximum(e1, 1-e0)

    conf=1-np.minimum(e0,e1)

    del e0, e1

    est_label = maximum_filter(est_label,(3,3))

    #print('Probability of land computed')

    # plt.imshow(bigimage); plt.imshow(est_label, alpha=0.5); plt.show()

    out_stack = np.dstack((est_label,conf,var0+var1))
    del est_label, conf, var0, var1


    #print('Probability stack computed')

    ####===============================================================
    thres_land = threshold_otsu(out_stack[:,:,0])
    thres_conf = threshold_otsu(out_stack[:,:,1])
    thres_var = threshold_otsu(out_stack[:,:,2])
    print("Land threshold: %f" % (thres_land))
    print("Confidence threshold: %f" % (thres_conf))
    print("Variance threshold: %f" % (thres_var))

    meta['otsu_land'] = thres_land
    meta['otsu_confidence'] = thres_conf
    meta['otsu_variance'] = thres_var


    #mask1 (conservative)
    mask1 = ((out_stack[:,:,0]>thres_land) & (out_stack[:,:,1]>thres_conf) & (out_stack[:,:,2]<thres_var)).astype('uint8')
    mask2 = ((out_stack[:,:,0]>thres_land) & (out_stack[:,:,2]<thres_var)).astype('uint8')
    mask3 = ((out_stack[:,:,0]>thres_land) & (out_stack[:,:,1]>thres_conf)).astype('uint8')
    #mask4 = (out_stack[:,:,0]>thres_land)
    mask4 = (out_stack[:,:,0]>thres_land).astype('uint8')

    #ultra-conservative
    mask0 = ((mask1+mask2+mask3+mask4)==4).astype('uint8')

	#ultra-liberal
    #mask5 = (out_stack[:,:,0]>thres_land) & (out_stack[:,:,1]<thres_conf) & (out_stack[:,:,2]>thres_var)

    # mask5 = ((mask1+mask2+mask3+mask4)>0).astype('uint8')
    # mask5[out_stack[:,:,1]<thres_conf] = 1

    # mask6 = (out_stack[:,:,0]>.1) & (out_stack[:,:,1]<.5)
    # mask6 = ((mask1+mask2+mask3+mask4+mask5+mask6)>0).astype('uint8')
	
    mask5 = mask4 + (out_stack[:,:,1]<thres_conf).astype('uint8')
    mask5[mask5>1]=1

    mask6 = mask5 + (out_stack[:,:,2]>thres_var).astype('uint8')
    mask6[mask6>1]=1

    #nx,ny=np.shape(mask1)
    #island_thres = 100*np.maximum(nx,ny)
    #meta['island_thres'] = island_thres

    # crf_theta_slider_value=1
    # crf_mu_slider_value=1
    # crf_downsample_factor=5
    # gt_prob=0.51
    # mask3 , n = crf_refine(mask0+1,bigimage,crf_theta_slider_value,crf_mu_slider_value,crf_downsample_factor,gt_prob)
    # mask4 , n = crf_refine(mask1+1,bigimage,crf_theta_slider_value,crf_mu_slider_value,crf_downsample_factor,gt_prob)
    # mask5 , n = crf_refine(mask2+1,bigimage,crf_theta_slider_value,crf_mu_slider_value,crf_downsample_factor,gt_prob)

    # mask3 -= 1
    # mask4 -= 1
    # mask5 -= 1

    # mask0 = remove_small_holes(mask0.astype('bool'), island_thres).astype('uint8')
    # mask1 = remove_small_holes(mask1.astype('bool'), island_thres).astype('uint8')
    # mask2 = remove_small_holes(mask2.astype('bool'), island_thres).astype('uint8')
    # mask3 = remove_small_holes(mask3.astype('bool'), island_thres).astype('uint8')
    # mask4 = remove_small_holes(mask4.astype('bool'), island_thres).astype('uint8')
    # mask5 = remove_small_holes(mask5.astype('bool'), island_thres).astype('uint8')

    mask6 = ((mask0+mask1+mask2+mask3+mask4+mask5)>0).astype('uint8')
    #mask6 = remove_small_holes(mask6.astype('bool'), island_thres).astype('uint8')

    print('Land masks computed')

    elapsed = (time.time() - start)/60
    meta['elapsed_minutes'] = elapsed

    print("Image masking took "+ str(elapsed) + " minutes")


    #====================
    #====================

    start = time.time()
		
    #====================

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
	
    del out_stack
	
    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'meta'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'meta'))
    except:
        pass
		
    np.savez_compressed(outfile.replace('.tif','.npz'), **meta)
	
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
    #del mask4

    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks5'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks5'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask5.astype('uint8'),quality=100)
    del mask5

    #====================
    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'masks6'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'masks6'))
    except:
        pass

    imsave(outfile.replace('.tif','.jpg'),255*mask6.astype('uint8'),quality=100)
    # del mask6

    #====================
    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(mask6, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(mask6, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

    # del land

    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'mask6_overlays'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'mask6_overlays'))
    except:
        pass

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=200, bbox_inches='tight')
    plt.close('all')

    #====================
    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(mask4, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(mask4, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)

    # del land

    outfile = segfile.replace(os.path.normpath(sample_direc), os.path.normpath(sample_direc+os.sep+'mask4_overlays'))

    try:
        os.mkdir(os.path.normpath(sample_direc+os.sep+'mask4_overlays'))
    except:
        pass

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=200, bbox_inches='tight')
    plt.close('all')


    del bigimage, color_label, mask4, mask6

    elapsed = (time.time() - start)/60

    print('Outputs made')
    print("Writing outputs took "+ str(elapsed) + " minutes")



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


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')


from imports import *

#=======================================================
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


M= []; C=[]; T = []
for counter,w in enumerate(weights):
    configfile = w.replace('.h5','.json').replace('weights', 'config')


    with open(configfile) as f:
        config = json.load(f)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')


    from imports import *

    #=======================================================

    print('.....................................')
    print('Creating and compiling model {}...'.format(counter))

    if MODEL =='resunet':
        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )
    elif MODEL=='unet':
        model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )

    elif MODEL =='simple_resunet':
        # num_filters = 8 # initial filters
        # model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_filters, NCLASSES, (KERNEL_SIZE, KERNEL_SIZE))

        model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #346,564
    elif MODEL=='simple_unet':
        model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))
    #242,812

    elif MODEL=='satunet':
        #model = sat_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS), num_classes=NCLASSES)

        model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel = (2, 2),
                    num_classes=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    activation="relu",
                    use_batch_norm=True,
                    dropout=DROPOUT,#0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                    dropout_type=DROPOUT_TYPE,#"standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                    filters=FILTERS,#8,
                    num_layers=4,
                    strides=(1,1))

    else:
        print("Model must be one of 'unet', 'resunet', or 'satunet'")
        sys.exit(2)


    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])
    model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

    model.load_weights(w)

    M.append(model)
    C.append(configfile)
    T.append(MODEL)


# metadatadict = {}
meta['model_weights'] = W
meta['config_files'] = C
meta['model_types'] = T

if 'WEIGHTING' in locals():
    meta['WEIGHTING'] = WEIGHTING
else:
    WEIGHTING =  [1 for m in M]
    meta['WEIGHTING'] = WEIGHTING

# W = [1 for m in M]

### predict
print('.....................................')
print('Using model for prediction on images ...')

sample_filenames = sorted(glob(sample_direc+os.sep+'*.jpg'))

print('Number of samples: %i' % (len(sample_filenames)))

for counter,f in enumerate(sample_filenames):
    do_seg(f, M, WEIGHTING, meta)
    print('%i out of %i done'%(counter,len(sample_filenames)))


#w = Parallel(n_jobs=-2, verbose=1, max_nbytes=None)(delayed(do_seg)(f,M, WEIGHTING, meta) for f in tqdm(sample_filenames))
