# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021-23, Marda Science LLC
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

import json
from doodleverse_utils.imports import *
# from scipy.ndimage import maximum_filter
from skimage.transform import resize
from skimage.filters import threshold_otsu
# from joblib import Parallel, delayed

#####################################################
#### definitions


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

    bigimage = imread(f)

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
def get_prob_stacks(M, image_stan, w, h):

    # w = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(run_model(image_stan, model))() for model in M)
    # E0, E1 = zip(*w)

    E0 = []; E1 = [];

    for model in M:

        e0, e1 = run_model(image_stan, model, w, h)

        E0.append(e0)
        E1.append(e1)
        # del e0, e1
    return E0, E1

# =========================================================
def run_model(image_stan, model, w, h):
    est_label = model(tf.expand_dims(image_stan, 0))
    est_label = tf.squeeze(est_label).numpy()

    e0 = resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True)
    e1 = resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True)
    # e1 = maximum_filter(e1,(3,3))

    return e0, e1


# =========================================================

def average_probstack(E0,E1,WEIGHTING):
    e0 = np.average(np.dstack(E0), axis=-1, weights=np.array(WEIGHTING))

    var0 = np.std(np.dstack(E0), axis=-1)

    e1 = np.average(np.dstack(E1), axis=-1, weights=np.array(WEIGHTING))

    var1 = np.std(np.dstack(E1), axis=-1)

    est_label = np.maximum(e1, 1-e0)

    conf=1-np.minimum(e0,e1)

    # est_label = maximum_filter(est_label,(3,3))

    out_stack = np.dstack((est_label,conf,var0+var1))
    return out_stack

# =========================================================
def return_img_array(f, TARGET_SIZE):
    if 'jpg' in f:
        segfile = f.replace('.jpg', '_seg.tif')
    elif 'png' in f:
        segfile = f.replace('.png', '_seg.tif')

    segfile = os.path.normpath(segfile)

    image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
    image_stan = standardize(image.numpy()).squeeze()
    return image_stan, w, h, segfile, bigimage

# =========================================================
def print_figs(segfile, bigimage, out_stack, mask6, out_folder, meta):

    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'prob_stack'))

    imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)


    # outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'probstack_overlays'))

    # plt.imshow(bigimage); plt.imshow(out_stack, alpha=0.5);
    # plt.axis('off')
    # # plt.show()
    # plt.savefig(outfile.replace('.tif','.jpg'), dpi=100, bbox_inches='tight')
    # plt.close('all')	
	
    # del out_stack

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'masks6'))

    imsave(outfile.replace('.tif','.jpg'),255*mask6.astype('uint8'),quality=100)

    #====================
    class_label_colormap = ['#3366CC','#DC3912']
    try:
        color_label = label_to_colors(mask6, bigimage.numpy()[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
    except:
        color_label = label_to_colors(mask6, bigimage[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)


    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'mask6_overlays'))

    plt.imshow(bigimage); plt.imshow(color_label, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=100, bbox_inches='tight')
    plt.close('all')

    #====================
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'meta'))
		
    np.savez_compressed(outfile.replace('.tif','.npz'), **meta)

    # gc.collect()


# =========================================================
def do_seg(f, Mc, WEIGHTING, TARGET_SIZE, meta, out_folder, logic=6):
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

    try:
        # Mc = []
        # for m in M:
        #     m.compile(optimizer='adam')
        #     Mc.append(m)

        meta['image_filename'] = f.split(os.sep)[-1]

        image_stan, w, h, segfile, bigimage = return_img_array(f, TARGET_SIZE)

        E0, E1 = get_prob_stacks(Mc, image_stan, w, h)

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

        print_figs(segfile, bigimage, out_stack, mask, out_folder, meta)

    except:
        print("{} failed".format(f))

    # gc.collect()
