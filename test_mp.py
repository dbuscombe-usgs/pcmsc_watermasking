
from glob import glob 

import os, time, sys, getopt, json, multiprocessing
import subprocess
from typing import Tuple
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from joblib import Parallel, delayed

from doodleverse_utils.imports import *
from scipy.ndimage import maximum_filter
from skimage.transform import resize
from skimage.filters import threshold_otsu

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)

#####################################################
#### definitions

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

    thres_land = threshold_otsu(out_stack[:,:,0])
    thres_conf = threshold_otsu(out_stack[:,:,1])
    thres_var = threshold_otsu(out_stack[:,:,2])

    return thres_land, thres_conf, thres_var

# =========================================================
def get_prob_stacks(M, image_stan, w, h):

    E0 = []; E1 = [];

    for model in M:

        # est_label = model.predict(tf.expand_dims(image_stan, 0) , batch_size=1, verbose = 0).squeeze()
        est_label = model(tf.expand_dims(image_stan, 0))
        est_label = tf.squeeze(est_label).numpy()

        e0 = resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True)
        e1 = resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True)
        e1 = maximum_filter(e1,(3,3))

        E0.append(e0)
        E1.append(e1)
        del est_label, e0, e1
    return E0, E1

# =========================================================

def average_probstack(E0,E1,WEIGHTING):
    e0 = np.average(np.dstack(E0), axis=-1, weights=np.array(WEIGHTING))

    var0 = np.std(np.dstack(E0), axis=-1)

    e1 = np.average(np.dstack(E1), axis=-1, weights=np.array(WEIGHTING))

    var1 = np.std(np.dstack(E1), axis=-1)

    est_label = np.maximum(e1, 1-e0)

    conf=1-np.minimum(e0,e1)

    est_label = maximum_filter(est_label,(3,3))

    out_stack = np.dstack((est_label,conf,var0+var1))
    return out_stack

# =========================================================
def return_img_array(f):
    if 'jpg' in f:
        segfile = f.replace('.jpg', '_seg.tif')
    elif 'png' in f:
        segfile = f.replace('.png', '_seg.tif')

    segfile = os.path.normpath(segfile)

    image, w, h, bigimage = seg_file2tensor_3band(f)
    image_stan = standardize(image.numpy()).squeeze()
    return image_stan, w, h, segfile, bigimage

# =========================================================
def print_figs(segfile, bigimage, out_stack, mask6, out_folder):

    start = time.time()
    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'prob_stack'))

    imsave(outfile.replace('.tif','.png'),(100*out_stack).astype('uint8'),compression=9)


    outfile = segfile.replace(os.path.dirname(segfile), os.path.normpath(out_folder+os.sep+'probstack_overlays'))

    plt.imshow(bigimage); plt.imshow(out_stack, alpha=0.5);
    plt.axis('off')
    # plt.show()
    plt.savefig(outfile.replace('.tif','.jpg'), dpi=100, bbox_inches='tight')
    plt.close('all')	
	
    del out_stack

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

    elapsed = (time.time() - start)/60
    print("Writing outputs took "+ str(elapsed) + " minutes")


# =========================================================
def do_seg(f, M, WEIGHTING, meta, out_folder):
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

    # for model in M:
    #     model.compile(optimizer = 'adam')

    # Mc = [model.compile(optimizer = 'adam') for model in M]

    Mc = []
    for m in M:
        m.compile(optimizer='adam')
        Mc.append(m)

    meta['image_filename'] = f.split(os.sep)[-1]

    start = time.time()

    image_stan, w, h, segfile, bigimage = return_img_array(f)

    E0, E1 = get_prob_stacks(Mc, image_stan, w, h)

    K.clear_session()
    del image_stan, Mc

    out_stack = average_probstack(E0,E1,WEIGHTING)
    del E0, E1

    thres_land, thres_conf, thres_var = get_thres(out_stack)
    
    meta['otsu_land'] = thres_land
    meta['otsu_confidence'] = thres_conf
    meta['otsu_variance'] = thres_var

    mask4 = (out_stack[:,:,0]>thres_land).astype('uint8')
	
    mask5 = mask4 + (out_stack[:,:,1]<thres_conf).astype('uint8')
    mask5[mask5>1]=1

    mask6 = mask5 + (out_stack[:,:,2]>thres_var).astype('uint8')
    mask6[mask6>1]=1

    elapsed = (time.time() - start)/60
    meta['elapsed_minutes'] = elapsed
    print("Image masking took "+ str(elapsed) + " minutes")

    print_figs(segfile, bigimage, out_stack, mask6, out_folder)



###=================================

out_folder = r"F:\watermasking_benchmark\CenCA_coastal_20160308\images"

configfile  = "F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_windows_4resunets.json"

sample_filenames = glob("F:\watermasking_benchmark\CenCA_coastal_20220609\images\*.JPG")


with open(configfile) as f:
    config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

#=======================================================
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

meta['TARGET_SIZE'] = TARGET_SIZE


M= []; C=[]; T = []; W = []
for counter,w in enumerate(weights):
    if 'fullmodel' in w:
        configfile = w.replace('_fullmodel.h5','.json').replace('weights', 'config')
    else:
        configfile = w.replace('.h5','.json').replace('weights', 'config')

    with open(configfile) as f:
        config = json.load(f)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')

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

    # model.compile(optimizer = 'adam') #, loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

    model.load_weights(w)

    M.append(model)
    C.append(configfile)
    T.append(MODEL)
    W.append(weights)


meta['model_weights'] = W
meta['config_files'] = C
meta['model_types'] = T

if 'WEIGHTING' in locals():
    meta['WEIGHTING'] = WEIGHTING
else:
    WEIGHTING =  [1 for m in M]
    meta['WEIGHTING'] = WEIGHTING


try:
    os.mkdir(os.path.normpath(out_folder+os.sep+'mask6_overlays'))
except:
    pass

try:
    os.mkdir(os.path.normpath(out_folder+os.sep+'masks6'))
except:
    pass

try:
    os.mkdir(os.path.normpath(out_folder+os.sep+'meta'))
except:
    pass

try:
    os.mkdir(os.path.normpath(out_folder+os.sep+'prob_stack'))
except:
    pass

try:
    os.mkdir(os.path.normpath(out_folder+os.sep+'probstack_overlays'))
except:
    pass


results = Parallel(n_jobs=-1, verbose=1, timeout=9999)(delayed(do_seg(f, M, WEIGHTING, meta, out_folder))() for f in sample_filenames if not os.path.exists(f.replace(os.path.dirname(f), out_folder+os.sep+"meta").replace('.jpg','_seg.npz') ))



# def test_func(k, M, WEIGHTING, meta, out_folder):
#     print(k)
#     time.sleep(5) # Pause
#     for model in M:
#         model.compile(optimizer = 'adam')


# Parallel(n_jobs=-1, verbose=1, timeout=99999)(delayed(test_func(k)) for k in sample_filenames)
# results = Parallel(n_jobs=-2, verbose=1, timeout=9999)(delayed(test_func(k, M,  WEIGHTING, meta, out_folder))() for k in sample_filenames)


    # with open(files, "r") as f:
    #     sample_filenames = f.readlines()

    # sample_filenames = [f.split('\n')[0] for f in sample_filenames]

    # for counter,f in enumerate(sample_filenames):

    #     check_file = f.replace(os.path.dirname(f), out_folder+os.sep+"meta").replace('.jpg','_seg.npz') 
        
    #     if not os.path.exists(check_file):
    #         do_seg(f, M, WEIGHTING, meta, out_folder)
    #     print('%i out of %i done'%(counter,len(sample_filenames)))

    # os.remove(files)

