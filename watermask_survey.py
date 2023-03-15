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
import random, string
from glob import glob


###########################################################################
##### MAIN

# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20160308\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20170125\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20180910\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20190223\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20200419\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20211218\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 
# python .\watermask_survey.py -f F:\watermasking_benchmark\CenCA_coastal_20220609\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json ; 

if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:f:c:o:")
    except getopt.GetoptError:
        print('python watermask_survey.py -f folder -c json -o out_folder \n')		
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python watermask_survey.py -f D:\for_watermasking\CenCA_coastal_20151209\images -c F:\dbuscombe_github\pcmsc_watermasking\config\watermask_benchmark_deploy_v2.json -o D:\watermasking_results\CenCA_coastal_20151209')
            sys.exit()

        elif opt in ("-f"):
            folder = arg
            folder = str(folder)
            folder = os.path.normpath(folder)

        elif opt in ("-c"):
            configfile = arg
            configfile = str(configfile)
            configfile = os.path.normpath(configfile)

        elif opt in ("-o"):
            out_folder = arg
            out_folder = str(out_folder)
            out_folder = os.path.normpath(out_folder)

    print(configfile)
    print(folder)
    print(out_folder)

    files = glob(folder+os.sep+'*.jpg')
    print("Found {} files in {}".format(len(files),folder))


    # write out sample filenames to temporary file
    tmp_file = ''.join(random.choices(string.ascii_letters + string.digits, k=7))+'.txt'

    with open(tmp_file, "w") as f:
        for k in files:
            f.write(k+'\n')

    # use subprocesses to call N instances of 'watermask.py', which will work on a list of filenames, t
    def call_watermask(t, out_folder, configfile):
        os.system('python watermask.py -f '+t+' -o '+out_folder+' -c '+configfile+' -r 0')

    call_watermask(tmp_file, out_folder, configfile)