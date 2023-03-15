# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2023, Marda Science LLC
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

#####################################################
#### imports

import os, sys, getopt, json, time
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from watermask_funcs import *

## python watermask_single_image.py -f F:\watermasking_benchmark\CenCA_coastal_20160308\images\CAM432_20160308214334_60.jpg -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json

if __name__ == '__main__':

    start_time = time.time()

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:f:c:o:")
    except getopt.GetoptError:
        print('python watermask_single_image.py -f jpg -c json -o folder\n')		
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python watermask_single_image.py -f F:\watermasking_benchmark\CenCA_coastal_20160308\images\CAM432_20160308214334_60.jpg -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json -o F:\watermasking_benchmark_results\CenCA_coastal_20160308')
            sys.exit()

        elif opt in ("-f"):
            f = arg
            f = str(f)
            f = os.path.normpath(f)

        elif opt in ("-c"):
            configfile = arg
            configfile = str(configfile)
            configfile = os.path.normpath(configfile)

        elif opt in ("-o"):
            out_folder = arg
            out_folder = str(out_folder)
            out_folder = os.path.normpath(out_folder)

    print(configfile)
    print(f)
    print(out_folder)
    # out_folder = os.path.dirname(f)

    try:
        os.mkdir(out_folder)
    except:
        pass

    ### read config deployment file
    with open(configfile) as cf:
        config = json.load(cf)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')

    ## compile model, make metadata
    meta = dict()

    MODEL = 'resunet'

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

    # Mc, meta = make_and_compile_model(meta, weights, MODEL)

    M= []; C=[]; T = []; W = []
    for counter,w in enumerate(weights):
        if 'fullmodel' in w:
            configfile = w.replace('_fullmodel.h5','.json').replace('weights', 'config')
        else:
            configfile = w.replace('.h5','.json').replace('weights', 'config')

        with open(configfile) as cf:
            config = json.load(cf)

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

        model.compile(optimizer = 'adam') #, loss = dice_coef_loss, metrics = [mean_iou, dice_coef])

        model.load_weights(w)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)
        W.append(weights)

    meta['model_weights'] = W
    meta['config_files'] = C
    meta['model_types'] = T
    del C, W, T

    if 'WEIGHTING' in locals():
        meta['WEIGHTING'] = WEIGHTING
    else:
        WEIGHTING =  [1 for m in M]
        meta['WEIGHTING'] = WEIGHTING

    # Mc = []
    # for m in M:
    #     m.compile(optimizer='adam')
    #     Mc.append(m)

    ## prep output folders

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

    print("--- Prep: %s seconds ---" % (time.time() - start_time))

    ## run watermasking
    do_seg(f, M, WEIGHTING, TARGET_SIZE, meta, out_folder)

    print("--- Overall: %s seconds ---" % (time.time() - start_time))
