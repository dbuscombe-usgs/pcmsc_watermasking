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

print('........................................')
print('ooooo...................................')
print('xxxxoo..................................')
print('xxxxxxo..............GENERATE...........')
print('xxxxxxxo................METASHAPE.......')
print('xxxxxxxxxxo.................INPUTS......')
print('xxxxxxxxxxxxxo..........................')
print('xxxxxxxxxxxxxxxxxxxxo...................')
print('xxxxxxxxxxxxxxxxxxxxxxxxxo..............')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxo...........')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxo..........')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo......')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo.....')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo...')
print('xxxxxxxxWESTCOASTxxxxxxxxxxxxxxxxxxxxo..')
print('xxxxxxxxxxx WATERMASKER xxxxxxxxxxxxxxo.')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxo')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

#####################################################
#### imports

import os, sys, getopt
import random, string, threading, multiprocessing
from glob import glob


#####################################################
#### definitions

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


###########################################################################
##### MAIN

## python .\watermask_survey_splitfolders.py -f F:\watermasking_benchmark\CenCA_coastal_20160308\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v3.json 

if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:f:c:")
    except getopt.GetoptError:
        print('python watermask_survey_splitfolders.py -f folder\n')		
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python watermask_survey_splitfolders.py -f D:\for_watermasking\CenCA_coastal_20151209\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_windows_4resunets.json')
            # python .\watermask_survey_splitfolders.py -f F:\watermasking_benchmark\CenCA_coastal_20220609\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_windows_4resunets.json
            sys.exit()

        elif opt in ("-f"):
            folder = arg
            folder = str(folder)
            folder = os.path.normpath(folder)

        elif opt in ("-c"):
            configfile = arg
            configfile = str(configfile)
            configfile = os.path.normpath(configfile)

    print(configfile)
    print(folder)

    files = glob(folder+os.sep+'*.jpg')
    print("Found {} files in {}".format(len(files),folder))

    # discover number of cores
    Ncores = multiprocessing.cpu_count()
    print("Found {} processors".format(Ncores))

    # split filenames into N sets
    chunks = list(split(files, Ncores))

    # create N folders for outputs
    out_folders = []
    try:
        for k in range(len(chunks)):
            os.mkdir(folder+os.sep+'folder'+str(k))
            out_folders.append(folder+os.sep+'folder'+str(k))
    except:
        for k in range(len(chunks)):
            out_folders.append(folder+os.sep+'folder'+str(k))

    # write out chunks to temp files and keep track using T
    T = []
    for counter, sample_filenames in enumerate(chunks):

        # write out sample filenames to temporary file
        tmp_file = ''.join(random.choices(string.ascii_letters + string.digits, k=7))+'.txt'

        with open(tmp_file, "w") as f:
            for k in sample_filenames:
                f.write(k+'\n')
        T.append(tmp_file)


    # use subprocesses to call N instances of 'watermask.py', which will work on a list of filenames, t
    def call_watermask(t, out_folder, configfile):
        os.system('python watermask.py -f '+t+' -o '+out_folder+' -c '+configfile)

    threads_list = []
    for k in range(len(T)):
        t1 = threading.Thread( target=call_watermask, args=(T[k],out_folders[k], configfile) )
        threads_list.append(t1)

    for t1 in threads_list:
        t1.start()

