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

import os, shutil, sys, getopt, json, time 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from watermask_funcs import *

###########################################################################
##### MAIN

if __name__ == '__main__':

    start_time = time.time()

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:f:o:c:r:")
    except getopt.GetoptError:
        print('python watermask.py -f files -o out_folder -c configfile [-r move_folders]')		
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python watermask.py -f my_files.txt -o F:\masks -c F:\congif_files\my_configfile.json -r 1')
            sys.exit()

        elif opt in ("-f"):
            files = arg
            files = str(files)
            files = os.path.normpath(files)

        elif opt in ("-o"):
            out_folder = arg
            out_folder = str(out_folder)
            out_folder = os.path.normpath(out_folder)

        elif opt in ("-c"):
            configfile = arg
            configfile = str(configfile)
            configfile = os.path.normpath(configfile)

        elif opt in ("-r"):
            move_folders = arg
            move_folders = int(move_folders)
            if move_folders == 1:
                move_folders = True
            else:
                move_folders = False

    if 'move_folders' not in locals():
        move_folders = True

    if move_folders:
        print("folders will be moved after")
    else:
        print("folders will not be moved after")


    with open(configfile) as f:
        config = json.load(f)

    for k in config.keys():
        exec(k+'=config["'+k+'"]')


    with open(files, "r") as f:
        sample_filenames = f.readlines()

    sample_filenames = [f.split('\n')[0] for f in sample_filenames]
    # print("Found {} files".format(len(sample_filenames)))

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

    for f in sample_filenames:
        do_seg(f, M, WEIGHTING, TARGET_SIZE, meta, out_folder)

    os.remove(files)

    if move_folders:
        dst_folder = os.sep.join(out_folder.split(os.sep)[:-1])

        shutil.move(out_folder+os.sep+"mask6_overlays", dst_folder)
        shutil.move(out_folder+os.sep+"masks6", dst_folder)
        shutil.move(out_folder+os.sep+"meta", dst_folder)
        shutil.move(out_folder+os.sep+"prob_stack", dst_folder)

        os.rmdir(out_folder)

    print("--- Overall: %s seconds ---" % (time.time() - start_time))

    # try:
    #     _ = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(do_seg(f, M, WEIGHTING, meta, out_folder))() for f in sample_filenames if not os.path.exists(f.replace(os.path.dirname(f), out_folder+os.sep+"meta").replace('.jpg','_seg.npz') ))
    # except:
    #     _ = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(do_seg(f, M, WEIGHTING, meta, out_folder))() for f in sample_filenames if not os.path.exists(f.replace(os.path.dirname(f), out_folder+os.sep+"meta").replace('.jpg','_seg.npz') ))
    # finally:
    #     _ = Parallel(n_jobs=-1, verbose=10, timeout=9999)(delayed(do_seg(f, M, WEIGHTING, meta, out_folder))() for f in sample_filenames if not os.path.exists(f.replace(os.path.dirname(f), out_folder+os.sep+"meta").replace('.jpg','_seg.npz') ))
